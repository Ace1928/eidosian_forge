import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
class Mask2FormerPredictionBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state