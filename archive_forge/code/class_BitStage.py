import collections
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig
class BitStage(nn.Module):
    """
    A ResNet v2 stage composed by stacked layers.
    """

    def __init__(self, config, in_channels, out_channels, stride, dilation, depth, bottle_ratio=0.25, layer_dropout=None):
        super().__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        if config.layer_type == 'bottleneck':
            layer_cls = BitBottleneckLayer
        else:
            layer_cls = BitPreActivationBottleneckLayer
        prev_chs = in_channels
        self.layers = nn.Sequential()
        for layer_idx in range(depth):
            stride, drop_path_rate, is_first_layer = self._get_updated_hyperparameters(layer_idx, stride, layer_dropout)
            self.layers.add_module(str(layer_idx), layer_cls(config, prev_chs, out_channels, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, first_dilation=first_dilation, drop_path_rate=drop_path_rate, is_first_layer=is_first_layer))
            prev_chs = out_channels
            first_dilation = dilation

    def _get_updated_hyperparameters(self, layer_idx, stride, layer_dropout):
        """
        Get the new hyper-parameters with respect to the previous ones and the index of the current layer.
        """
        if layer_dropout:
            drop_path_rate = layer_dropout[layer_idx]
        else:
            drop_path_rate = 0.0
        if layer_idx != 0:
            stride = 1
        is_first_layer = layer_idx == 0
        return (stride, drop_path_rate, is_first_layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for _, layer in enumerate(self.layers):
            hidden_state = layer(hidden_state)
        return hidden_state