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
class MobileViTV2DeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__()
        self.aspp = MobileViTV2ASPP(config)
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)
        self.classifier = MobileViTV2ConvLayer(config, in_channels=config.aspp_out_channels, out_channels=config.num_labels, kernel_size=1, use_normalization=False, use_activation=False, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = self.aspp(hidden_states[-1])
        features = self.dropout(features)
        features = self.classifier(features)
        return features