from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
class MobileViTV2TransformerLayer(nn.Module):

    def __init__(self, config: MobileViTV2Config, embed_dim: int, ffn_latent_dim: int, dropout: float=0.0) -> None:
        super().__init__()
        self.layernorm_before = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)
        self.attention = MobileViTV2LinearSelfAttention(config, embed_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm_after = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)
        self.ffn = MobileViTV2FFN(config, embed_dim, ffn_latent_dim, config.ffn_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layernorm_1_out = self.layernorm_before(hidden_states)
        attention_output = self.attention(layernorm_1_out)
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.ffn(layer_output)
        layer_output = layer_output + hidden_states
        return layer_output