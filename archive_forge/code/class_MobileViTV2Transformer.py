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
class MobileViTV2Transformer(nn.Module):

    def __init__(self, config: MobileViTV2Config, n_layers: int, d_model: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multiplier
        ffn_dims = [ffn_multiplier * d_model] * n_layers
        ffn_dims = [int(d // 16 * 16) for d in ffn_dims]
        self.layer = nn.ModuleList()
        for block_idx in range(n_layers):
            transformer_layer = MobileViTV2TransformerLayer(config, embed_dim=d_model, ffn_latent_dim=ffn_dims[block_idx])
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states