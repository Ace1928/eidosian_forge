import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinat import DinatConfig
class DinatLayer(nn.Module):

    def __init__(self, config, dim, num_heads, dilation, drop_path_rate=0.0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.kernel_size = config.kernel_size
        self.dilation = dilation
        self.window_size = self.kernel_size * self.dilation
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = NeighborhoodAttentionModule(config, dim, num_heads, kernel_size=self.kernel_size, dilation=self.dilation)
        self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = DinatIntermediate(config, dim)
        self.output = DinatOutput(config, dim)
        self.layer_scale_parameters = nn.Parameter(config.layer_scale_init_value * torch.ones((2, dim)), requires_grad=True) if config.layer_scale_init_value > 0 else None

    def maybe_pad(self, hidden_states, height, width):
        window_size = self.window_size
        pad_values = (0, 0, 0, 0, 0, 0)
        if height < window_size or width < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - width)
            pad_b = max(0, window_size - height)
            pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return (hidden_states, pad_values)

    def forward(self, hidden_states: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width, channels = hidden_states.size()
        shortcut = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_output = attention_output[:, :height, :width, :].contiguous()
        if self.layer_scale_parameters is not None:
            attention_output = self.layer_scale_parameters[0] * attention_output
        hidden_states = shortcut + self.drop_path(attention_output)
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.output(self.intermediate(layer_output))
        if self.layer_scale_parameters is not None:
            layer_output = self.layer_scale_parameters[1] * layer_output
        layer_output = hidden_states + self.drop_path(layer_output)
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs