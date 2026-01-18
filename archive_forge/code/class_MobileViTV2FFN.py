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
class MobileViTV2FFN(nn.Module):

    def __init__(self, config: MobileViTV2Config, embed_dim: int, ffn_latent_dim: int, ffn_dropout: float=0.0) -> None:
        super().__init__()
        self.conv1 = MobileViTV2ConvLayer(config=config, in_channels=embed_dim, out_channels=ffn_latent_dim, kernel_size=1, stride=1, bias=True, use_normalization=False, use_activation=True)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.conv2 = MobileViTV2ConvLayer(config=config, in_channels=ffn_latent_dim, out_channels=embed_dim, kernel_size=1, stride=1, bias=True, use_normalization=False, use_activation=False)
        self.dropout2 = nn.Dropout(ffn_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states