import math
from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import MLP, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool=True, proj_bias: bool=True, attention_dropout: float=0.0, dropout: float=0.0):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError('window_size and shift_size must be of length 2')
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer('relative_position_index', relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(self.relative_position_bias_table, self.relative_position_index, self.window_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(x, self.qkv.weight, self.proj.weight, relative_position_bias, self.window_size, self.num_heads, shift_size=self.shift_size, attention_dropout=self.attention_dropout, dropout=self.dropout, qkv_bias=self.qkv.bias, proj_bias=self.proj.bias, training=self.training)