from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_outputs import SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import load_backbone
from .configuration_upernet import UperNetConfig
class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int], str]=0, bias: bool=False, dilation: Union[int, Tuple[int, int]]=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias, dilation=dilation)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)
        return output