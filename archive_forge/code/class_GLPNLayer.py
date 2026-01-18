import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_glpn import GLPNConfig
class GLPNLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = GLPNAttention(config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequence_reduction_ratio=sequence_reduction_ratio)
        self.drop_path = GLPNDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = GLPNMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_attention_outputs = self.attention(self.layer_norm_1(hidden_states), height, width, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states
        outputs = (layer_output,) + outputs
        return outputs