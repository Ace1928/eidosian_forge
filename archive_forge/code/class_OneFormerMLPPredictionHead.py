import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerMLPPredictionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(PredictionBlock(in_dim, out_dim, activation=nn.ReLU() if i < num_layers - 1 else nn.Identity()))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)