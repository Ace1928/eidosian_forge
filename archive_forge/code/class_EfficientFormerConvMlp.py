import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerConvMlp(nn.Module):

    def __init__(self, config: EfficientFormerConfig, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, drop: float=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.convolution1 = nn.Conv2d(in_features, hidden_features, 1)
        self.activation = ACT2FN[config.hidden_act]
        self.convolution2 = nn.Conv2d(hidden_features, out_features, 1)
        self.dropout = nn.Dropout(drop)
        self.batchnorm_before = nn.BatchNorm2d(hidden_features, eps=config.batch_norm_eps)
        self.batchnorm_after = nn.BatchNorm2d(out_features, eps=config.batch_norm_eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.batchnorm_before(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        hidden_state = self.batchnorm_after(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state