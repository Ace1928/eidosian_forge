import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config
class PvtV2BlockLayer(nn.Module):

    def __init__(self, config: PvtV2Config, layer_idx: int, drop_path: float=0.0):
        super().__init__()
        hidden_size: int = config.hidden_sizes[layer_idx]
        num_attention_heads: int = config.num_attention_heads[layer_idx]
        spatial_reduction_ratio: int = config.sr_ratios[layer_idx]
        mlp_ratio: float = config.mlp_ratios[layer_idx]
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = PvtV2SelfAttention(config=config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, spatial_reduction_ratio=spatial_reduction_ratio)
        self.drop_path = PvtV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = PvtV2ConvFeedForwardNetwork(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool=False):
        self_attention_outputs = self.attention(hidden_states=self.layer_norm_1(hidden_states), height=height, width=width, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)
        mlp_output = self.drop_path(mlp_output)
        layer_output = hidden_states + mlp_output
        outputs = (layer_output,) + outputs
        return outputs