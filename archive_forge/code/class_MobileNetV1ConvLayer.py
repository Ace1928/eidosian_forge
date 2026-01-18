from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config
class MobileNetV1ConvLayer(nn.Module):

    def __init__(self, config: MobileNetV1Config, in_channels: int, out_channels: int, kernel_size: int, stride: Optional[int]=1, groups: Optional[int]=1, bias: bool=False, use_normalization: Optional[bool]=True, use_activation: Optional[bool or str]=True) -> None:
        super().__init__()
        self.config = config
        if in_channels % groups != 0:
            raise ValueError(f'Input channels ({in_channels}) are not divisible by {groups} groups.')
        if out_channels % groups != 0:
            raise ValueError(f'Output channels ({out_channels}) are not divisible by {groups} groups.')
        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, padding_mode='zeros')
        if use_normalization:
            self.normalization = nn.BatchNorm2d(num_features=out_channels, eps=config.layer_norm_eps, momentum=0.9997, affine=True, track_running_stats=True)
        else:
            self.normalization = None
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features