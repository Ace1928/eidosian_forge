import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxBottleneckBlock(nn.Module):

    def __init__(self, config: JukeboxVQVAEConfig):
        super().__init__()
        self.nb_discrete_codes = config.nb_discrete_codes
        self.codebook_width = config.embed_dim
        self.mu = config.lmu
        self.threshold = 1.0
        self.init = False
        self.codebook_sum = None
        self.codebook_elem = None
        self.register_buffer('codebook', torch.zeros(self.nb_discrete_codes, self.codebook_width))

    def _tile(self, hidden_states):
        dim, embed_width = hidden_states.shape
        if dim < self.nb_discrete_codes:
            n_repeats = (self.nb_discrete_codes + dim - 1) // dim
            std = 0.01 / np.sqrt(embed_width)
            hidden_states = hidden_states.repeat(n_repeats, 1)
            hidden_states = hidden_states + torch.randn_like(hidden_states) * std
        return hidden_states

    def init_codebook(self, hidden_states):
        nb_discrete_codes = self.nb_discrete_codes
        self.init = True
        codes = self._tile(hidden_states)
        self.codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
        self.codebook_sum = self.codebook
        self.codebook_elem = torch.ones(nb_discrete_codes, device=self.codebook.device)

    def update_codebook(self, hidden_states, latent_states):
        mu, codebook_width, nb_discrete_codes = (self.mu, self.codebook_width, self.nb_discrete_codes)
        with torch.no_grad():
            latent_states_onehot = torch.zeros(nb_discrete_codes, hidden_states.shape[0], device=hidden_states.device)
            latent_states_onehot.scatter_(0, latent_states.view(1, hidden_states.shape[0]), 1)
            _codebook_sum = torch.matmul(latent_states_onehot, hidden_states)
            _codebook_elem = latent_states_onehot.sum(dim=-1)
            codes = self._tile(hidden_states)
            _random_codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
            old_codebook = self.codebook
            self.codebook_sum = mu * self.codebook_sum + (1.0 - mu) * _codebook_sum
            self.codebook_elem = mu * self.codebook_elem + (1.0 - mu) * _codebook_elem
            usage = (self.codebook_elem.view(nb_discrete_codes, 1) >= self.threshold).float()
            norm_code = self.codebook_sum.view(nb_discrete_codes, codebook_width) / self.codebook_elem.view(nb_discrete_codes, 1)
            self.codebook = usage * norm_code + (1 - usage) * _random_codebook
            _codebook_prob = _codebook_elem / torch.sum(_codebook_elem)
            entropy = -torch.sum(_codebook_prob * torch.log(_codebook_prob + 1e-08))
            used_curr = (_codebook_elem >= self.threshold).sum()
            usage = torch.sum(usage)
            dk = torch.norm(self.codebook - old_codebook) / np.sqrt(np.prod(old_codebook.shape))
        return {'entropy': entropy, 'used_curr': used_curr, 'usage': usage, 'dk': dk}

    def preprocess(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if hidden_states.shape[-1] == self.codebook_width:
            prenorm = torch.norm(hidden_states - torch.mean(hidden_states)) / np.sqrt(np.prod(hidden_states.shape))
        elif hidden_states.shape[-1] == 2 * self.codebook_width:
            x1, x2 = (hidden_states[..., :self.codebook_width], hidden_states[..., self.codebook_width:])
            prenorm = torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape)) + torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            hidden_states = x1 + x2
        return (hidden_states, prenorm)

    def postprocess(self, latent_states, dequantised_states, x_shape):
        batch_size, time = x_shape
        dequantised_states = dequantised_states.view(batch_size, time, -1).permute(0, 2, 1).contiguous()
        latent_states = latent_states.view(batch_size, time)
        return (latent_states, dequantised_states)

    def quantise(self, latent_states):
        codebook_weights = self.codebook.t()
        distance = torch.sum(latent_states ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(latent_states, codebook_weights) + torch.sum(codebook_weights ** 2, dim=0, keepdim=True)
        min_distance, music_tokens = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return (music_tokens, fit)

    def dequantise(self, music_tokens):
        dequantised_states = F.embedding(music_tokens, self.codebook)
        return dequantised_states

    def encode(self, latent_states):
        samples, _, seq_len = latent_states.shape
        latent_states, _ = self.preprocess(latent_states)
        music_tokens, _ = self.quantise(latent_states)
        music_tokens = music_tokens.view(samples, seq_len)
        return music_tokens

    def decode(self, music_tokens):
        samples, seq_len = music_tokens.shape
        dequantised_states = self.dequantise(music_tokens)
        dequantised_states = dequantised_states.view(samples, seq_len, self.codebook_width).permute(0, 2, 1).contiguous()
        return dequantised_states

    def forward(self, hidden_states, update_codebook=True):
        samples, _, seq_len = hidden_states.shape
        hidden_states, prenorm = self.preprocess(hidden_states)
        if update_codebook and (not self.init):
            self.init_codebook(hidden_states)
        music_tokens, fit = self.quantise(hidden_states)
        dequantised_states = self.dequantise(music_tokens)
        if update_codebook:
            update_metrics = self.update_codebook(hidden_states, music_tokens)
        else:
            update_metrics = {}
        commit_loss = torch.norm(dequantised_states.detach() - hidden_states) ** 2 / np.prod(hidden_states.shape)
        dequantised_states = hidden_states + (dequantised_states - hidden_states).detach()
        music_tokens, dequantised_states = self.postprocess(music_tokens, dequantised_states, (samples, seq_len))
        return (music_tokens, dequantised_states, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics))