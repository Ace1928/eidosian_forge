from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig
class RegNetStage(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=2, depth: int=2):
        super().__init__()
        layer = RegNetXLayer if config.layer_type == 'x' else RegNetYLayer
        self.layers = nn.Sequential(layer(config, in_channels, out_channels, stride=stride), *[layer(config, out_channels, out_channels) for _ in range(depth - 1)])

    def forward(self, hidden_state):
        hidden_state = self.layers(hidden_state)
        return hidden_state