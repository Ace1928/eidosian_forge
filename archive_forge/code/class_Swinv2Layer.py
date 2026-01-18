import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_swinv2 import Swinv2Config
class Swinv2Layer(nn.Module):

    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, pretrained_window_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        window_size, shift_size = self._compute_window_shift((config.window_size, config.window_size), (shift_size, shift_size))
        self.window_size = window_size[0]
        self.shift_size = shift_size[0]
        self.attention = Swinv2Attention(config=config, dim=dim, num_heads=num_heads, window_size=self.window_size, pretrained_window_size=pretrained_window_size if isinstance(pretrained_window_size, collections.abc.Iterable) else (pretrained_window_size, pretrained_window_size))
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swinv2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return (window_size, shift_size)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            width_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return (hidden_states, pad_values)

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)
        attention_outputs = self.attention(hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()
        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.layernorm_before(attention_windows)
        hidden_states = shortcut + self.drop_path(hidden_states)
        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs