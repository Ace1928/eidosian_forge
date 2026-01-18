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
class MobileViTASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels
        if len(config.atrous_rates) != 3:
            raise ValueError('Expected 3 values for atrous_rates')
        self.convs = nn.ModuleList()
        in_projection = MobileViTConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=1, use_activation='relu')
        self.convs.append(in_projection)
        self.convs.extend([MobileViTConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=rate, use_activation='relu') for rate in config.atrous_rates])
        pool_layer = MobileViTASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)
        self.project = MobileViTConvLayer(config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation='relu')
        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pyramid = []
        for conv in self.convs:
            pyramid.append(conv(features))
        pyramid = torch.cat(pyramid, dim=1)
        pooled_features = self.project(pyramid)
        pooled_features = self.dropout(pooled_features)
        return pooled_features