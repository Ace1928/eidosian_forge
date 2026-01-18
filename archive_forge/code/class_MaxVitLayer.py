import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
class MaxVitLayer(nn.Module):
    """
    MaxVit layer consisting of a MBConv layer followed by a PartitionAttentionLayer with `window` and a PartitionAttentionLayer with `grid`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        grid_size (Tuple[int, int]): Size of the input feature grid.
    """

    def __init__(self, in_channels: int, out_channels: int, squeeze_ratio: float, expansion_ratio: float, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], head_dim: int, mlp_ratio: int, mlp_dropout: float, attention_dropout: float, p_stochastic_dropout: float, partition_size: int, grid_size: Tuple[int, int]) -> None:
        super().__init__()
        layers: OrderedDict = OrderedDict()
        layers['MBconv'] = MBConv(in_channels=in_channels, out_channels=out_channels, expansion_ratio=expansion_ratio, squeeze_ratio=squeeze_ratio, stride=stride, activation_layer=activation_layer, norm_layer=norm_layer, p_stochastic_dropout=p_stochastic_dropout)
        layers['window_attention'] = PartitionAttentionLayer(in_channels=out_channels, head_dim=head_dim, partition_size=partition_size, partition_type='window', grid_size=grid_size, mlp_ratio=mlp_ratio, activation_layer=activation_layer, norm_layer=nn.LayerNorm, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout, p_stochastic_dropout=p_stochastic_dropout)
        layers['grid_attention'] = PartitionAttentionLayer(in_channels=out_channels, head_dim=head_dim, partition_size=partition_size, partition_type='grid', grid_size=grid_size, mlp_ratio=mlp_ratio, activation_layer=activation_layer, norm_layer=nn.LayerNorm, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout, p_stochastic_dropout=p_stochastic_dropout)
        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        x = self.layers(x)
        return x