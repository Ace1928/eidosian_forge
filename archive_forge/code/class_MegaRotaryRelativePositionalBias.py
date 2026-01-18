import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaRotaryRelativePositionalBias(nn.Module):
    """
    Rotary relative bias for positional information; similar in concept to RoPE (i.e. RoFormer) but taken from the Mega
    repo due to differences in implementation.

    When initialized, produces a positional bias which ranges from position 0 to config.max_positions, but can
    extrapolate to longer sequences. Can be indexed according to input position IDs
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        if config.hidden_size % 2 != 0:
            raise RuntimeError('Rotary positional bias requires `hidden_size` to be a multiple of 2')
        self.config = config
        self.embed_dim = config.shared_representation_size
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(config.max_positions, self.embed_dim)
        self.alpha = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.b_param = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.register_buffer('_float_tensor', torch.FloatTensor([0.0]))

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return (torch.sin(emb), torch.cos(emb))

    def rotary(self, input):
        seq_len, embed_dim = input.size()
        chunk_1, chunk_2 = torch.chunk(input, 2, dim=-1)
        if self.sine is None or seq_len > self.sine.size(0):
            self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(seq_len, embed_dim)
            self.max_positions = seq_len
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)
        sin = self.sine[:seq_len]
        cos = self.cosine[:seq_len]
        return torch.cat([chunk_1 * cos - chunk_2 * sin, chunk_2 * cos + chunk_1 * sin], dim=1)

    def forward(self, seq_len):
        rotary_alpha = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        rotary_beta = self.rotary(self.b_param.expand(seq_len, self.embed_dim))
        bias = torch.einsum('mk,nk->mn', rotary_alpha, rotary_beta)
        return bias