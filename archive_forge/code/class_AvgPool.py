import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
from xformers.components.attention.utils import (
class AvgPool(nn.Module):

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        head_dim = x.shape[2]
        segments = seq_len // self.n
        assert segments > 0, 'num_landmarks should be smaller than the sequence length'
        if seq_len % self.n == 0:
            return x.reshape(-1, self.n, segments, head_dim).mean(dim=-2)
        n_round = self.n - seq_len % self.n
        x_avg_round = x[:, :n_round * segments, :].reshape(-1, n_round, segments, head_dim).mean(dim=-2)
        x_avg_off = x[:, n_round * segments:, :].reshape(-1, self.n - n_round, segments + 1, head_dim).mean(dim=-2)
        return torch.cat((x_avg_round, x_avg_off), dim=-2)