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
class JukeboxConditionalAutoregressive(nn.Module):

    def __init__(self, config, n_ctx=None, embed_dim=None, audio_conditioning=False, metadata_conditioning=False, is_encoder=False):
        """
        Autoregressive model on either lyric tokens or music tokens, or both. The attention pattern should be properly
        set fro each configuration.

        Args:
            config (`JukeboxPriorConfig`):
                Model configuration class with all the parameters of the model. Initializing with a config file does
                not load the weights associated with the model, only the configuration. Check out the
                [`~PreTrainedModel.from_pretrained`] method to load the model weights.
            n_ctx (`int`, *optional*):
                Number of tokens or lyrics tokens provided in a single pass.
            embed_dim (`int`, *optional*):
                Either equals to the dimension of the codebook, or the sum of n_vocab (lyrics) and codeboook dimension,
                if the model combines lyrics and music tokens, or simply n_vocab if the model is a seperate encoder
            audio_conditioning (`bool`, *optional*, defaults to `False`):
                Whether or not the prior supports conditionning on audio.
            metadata_conditioning (`bool`, *optional*, defaults to `False`):
                Whether or not the prior supports conditionning on artitst, genres, lyrics and timing.
            is_encoder (`bool`, *optional*, defaults to `False`):
                Whether the model is an encoder only model.
        """
        super().__init__()
        self.width = config.hidden_size
        self.num_layers = config.num_layers
        self.n_ctx = n_ctx if n_ctx is not None else config.n_ctx
        self.embed_dim = embed_dim if embed_dim is not None else config.music_vocab_size
        self.embed_tokens = nn.Embedding(self.embed_dim, config.hidden_size)
        self.embed_tokens_dropout = nn.Dropout(config.emb_dropout)
        self.metadata_conditioning = metadata_conditioning
        self.audio_conditioning = audio_conditioning
        if not metadata_conditioning:
            self.start_token = nn.Parameter(torch.empty((1, config.hidden_size)))
        self.pos_emb = JukeboxPositionalEmbedding(self.n_ctx, config.hidden_size)
        self.pos_emb_dropout = nn.Dropout(config.emb_dropout)
        self.transformer = JukeboxLayerStack(config, n_ctx=self.n_ctx)
        self.is_encoder = is_encoder
        self.encoder_len = config.nb_relevant_lyric_tokens
        if config.merged_decoder:
            self.add_cond_after_transformer = False
            self.share_embed_tokens_fc_proj_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_embed_tokens_fc_proj_out = True
        if not is_encoder:
            self.fc_proj_out = nn.Linear(config.hidden_size, self.embed_dim, bias=False)
            if self.share_embed_tokens_fc_proj_out:
                self.fc_proj_out.weight = self.embed_tokens.weight
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, audio_conditioning=None, metadata_conditioning=None, last_encoder_hidden_states=None, get_preds=False, get_acts=False, get_sep_loss=False):
        """
        Args:
            tokens (`torch.tensor`):
                Can represent music tokens, lyrics tokens or both, depending on the configuration.
        """
        batch_size = tokens.shape[0]
        with torch.no_grad():
            tokens = tokens.view(batch_size, -1).long()
        if not self.audio_conditioning:
            audio_conditioning = torch.zeros((batch_size, 1, self.width), device=tokens.device, dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype)
        target = tokens
        hidden_states = self.embed_tokens(tokens)
        hidden_states = torch.cat((hidden_states[:, -1:], hidden_states[:, :-1]), dim=1)
        if self.metadata_conditioning:
            hidden_states[:, 0] = metadata_conditioning.view(batch_size, self.width)
        else:
            hidden_states[:, 0] = self.start_token
        hidden_states = self.embed_tokens_dropout(hidden_states) + self.pos_emb_dropout(self.pos_emb()) + audio_conditioning
        hidden_states = self.transformer(hidden_states, last_encoder_hidden_states=last_encoder_hidden_states)
        if self.add_cond_after_transformer:
            hidden_states = hidden_states + audio_conditioning
        activations = hidden_states
        if self.is_encoder:
            return hidden_states
        hidden_states = self.fc_proj_out(hidden_states)
        loss_fn = nn.CrossEntropyLoss()
        if get_sep_loss:
            lyric_hidden_states = hidden_states[:, :self.encoder_len].reshape(-1, self.embed_dim)
            token_hidden_states = hidden_states[:, self.encoder_len:].reshape(-1, self.embed_dim)
            lyric_loss = loss_fn(lyric_hidden_states, target[:, :self.encoder_len].reshape(-1)) / np.log(2.0)
            music_token_loss = loss_fn(token_hidden_states, target[:, self.encoder_len:].reshape(-1)) / np.log(2.0)
            loss = (lyric_loss, music_token_loss)
        else:
            loss = loss_fn(hidden_states.view(-1, self.embed_dim), target.view(-1)) / np.log(2.0)
        if get_preds:
            return (loss, hidden_states)
        elif get_acts:
            return (loss, activations)
        else:
            return (loss, None)

    def get_emb(self, sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning):
        if sample_t == 0:
            hidden_states = torch.empty(n_samples, 1, self.width, dtype=self.embed_tokens.weight.dtype).to(self.embed_tokens.weight.device)
            if self.metadata_conditioning:
                hidden_states[:, 0] = metadata_conditioning.view(n_samples, self.width)
            else:
                hidden_states[:, 0] = self.start_token
        else:
            hidden_states = self.embed_tokens(tokens)
        if audio_conditioning.shape == (n_samples, self.n_ctx, self.width):
            cond = audio_conditioning[:, sample_t:sample_t + 1, :]
        else:
            cond = audio_conditioning
        hidden_states = hidden_states + self.pos_emb()[sample_t:sample_t + 1] + cond
        return (hidden_states, cond)

    def sample(self, n_samples, audio_conditioning=None, metadata_conditioning=None, last_encoder_hidden_states=None, temp=1.0, top_k=0, top_p=0.0, get_preds=False, sample_tokens=None):
        if sample_tokens is None:
            sample_tokens = self.n_ctx
        if not self.audio_conditioning:
            audio_conditioning = torch.zeros((n_samples, 1, self.width), dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype).to(self.fc_proj_out.device)
        with torch.no_grad():
            sampled_tokens = []
            tokens = None
            if get_preds:
                preds = []
            iter = tqdm(range(0, sample_tokens), leave=False)
            for sample_t in iter:
                iter.set_description(f'Ancestral sampling {sample_tokens} music tokens', refresh=True)
                hidden_states, cond = self.get_emb(sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning)
                hidden_states = self.transformer(hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=True)
                if self.add_cond_after_transformer:
                    hidden_states = hidden_states + cond
                hidden_states = self.fc_proj_out(hidden_states)
                if get_preds:
                    preds.append(hidden_states.clone())
                hidden_states = hidden_states / temp
                hidden_states = filter_logits(hidden_states, top_k=top_k, top_p=top_p)
                tokens = torch.distributions.Categorical(logits=hidden_states).sample()
                sampled_tokens.append(tokens.clone())
            del tokens
            self.transformer.del_cache()
            tokens = torch.cat(sampled_tokens, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
        if get_preds:
            return (tokens, preds)
        else:
            return tokens

    def split_chunks(self, length, chunk_size):
        n_passes = (length + chunk_size - 1) // chunk_size
        chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
        return chunk_sizes

    def primed_sample(self, n_samples, lyric_and_music_tokens, audio_conditioning=None, metadata_conditioning=None, last_encoder_hidden_states=None, temp=1.0, top_k=0, top_p=0.0, get_preds=False, chunk_size=None, sample_tokens=None):
        if sample_tokens is None:
            sample_tokens = self.n_ctx
        batch_size = lyric_and_music_tokens.shape[0]
        with torch.no_grad():
            lyric_and_music_tokens = lyric_and_music_tokens.view(batch_size, -1).long()
        sampled_audio = torch.split(lyric_and_music_tokens, 1, dim=1)
        sampled_audio = list(sampled_audio)
        if not self.audio_conditioning:
            audio_conditioning = torch.zeros((n_samples, 1, self.width), dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype).to(lyric_and_music_tokens.device)
        with torch.no_grad():
            if get_preds:
                preds = []
            if chunk_size is None:
                chunk_size = len(sampled_audio)
            chunk_sizes = self.split_chunks(len(sampled_audio), chunk_size)
            x_primes = []
            start = 0
            token = None
            for current_chunk_size in tqdm(chunk_sizes, desc='Preparing past key value', leave=False):
                sampled_audio_prime, conds_prime = ([], [])
                for sample_t in range(start, start + current_chunk_size):
                    x_prime, cond_prime = self.get_emb(sample_t, n_samples, token, audio_conditioning, metadata_conditioning)
                    token = sampled_audio[sample_t]
                    sampled_audio_prime.append(x_prime)
                    conds_prime.append(cond_prime)
                start = start + current_chunk_size
                x_prime, cond_prime = (torch.cat(sampled_audio_prime, dim=1), torch.cat(conds_prime, dim=1))
                del sampled_audio_prime
                del conds_prime
                if not get_preds:
                    del cond_prime
                x_prime = self.transformer(x_prime, last_encoder_hidden_states=last_encoder_hidden_states, sample=True)
                if get_preds:
                    if self.add_cond_after_transformer:
                        x_prime = x_prime + cond_prime
                    del cond_prime
                    x_primes.append(x_prime)
                else:
                    del x_prime
            if get_preds:
                x_prime = torch.cat(x_primes, dim=1)
                x_prime = self.fc_proj_out(x_prime)
                preds.append(x_prime)
            input_tokens = sampled_audio[-1]
            itererator = tqdm(range(len(sampled_audio), sample_tokens), desc=f'Sampling {len(range(len(sampled_audio), sample_tokens))} music tokens', leave=False)
            for sample_t in itererator:
                hidden_states, cond = self.get_emb(sample_t, n_samples, input_tokens, audio_conditioning, metadata_conditioning)
                hidden_states = self.transformer(hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=True)
                if self.add_cond_after_transformer:
                    hidden_states = hidden_states + cond
                hidden_states = self.fc_proj_out(hidden_states)
                if get_preds:
                    preds.append(hidden_states)
                hidden_states = hidden_states / temp
                hidden_states = filter_logits(hidden_states, top_k=top_k, top_p=top_p)
                music_tokens = torch.distributions.Categorical(logits=hidden_states).sample()
                sampled_audio.append(music_tokens.clone())
                input_tokens = music_tokens
            del input_tokens, music_tokens
            self.transformer.del_cache()
            music_tokens = torch.cat(sampled_audio, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
        if get_preds:
            return (music_tokens, preds)
        else:
            return music_tokens