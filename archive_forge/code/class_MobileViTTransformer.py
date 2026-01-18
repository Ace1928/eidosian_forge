import math
from typing import Dict, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilevit import MobileViTConfig
class MobileViTTransformer(nn.Module):

    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList()
        for _ in range(num_stages):
            transformer_layer = MobileViTTransformerLayer(config, hidden_size=hidden_size, intermediate_size=int(hidden_size * config.mlp_ratio))
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states