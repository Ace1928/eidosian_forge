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
class MaxVitBlock(nn.Module):
    """
    A MaxVit block consisting of `n_layers` MaxVit layers.

     Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        input_grid_size (Tuple[int, int]): Size of the input feature grid.
        n_layers (int): Number of layers in the block.
        p_stochastic (List[float]): List of probabilities for stochastic depth for each layer.
    """

    def __init__(self, in_channels: int, out_channels: int, squeeze_ratio: float, expansion_ratio: float, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], head_dim: int, mlp_ratio: int, mlp_dropout: float, attention_dropout: float, partition_size: int, input_grid_size: Tuple[int, int], n_layers: int, p_stochastic: List[float]) -> None:
        super().__init__()
        if not len(p_stochastic) == n_layers:
            raise ValueError(f'p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}.')
        self.layers = nn.ModuleList()
        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=3, stride=2, padding=1)
        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers += [MaxVitLayer(in_channels=in_channels if idx == 0 else out_channels, out_channels=out_channels, squeeze_ratio=squeeze_ratio, expansion_ratio=expansion_ratio, stride=stride, norm_layer=norm_layer, activation_layer=activation_layer, head_dim=head_dim, mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout, attention_dropout=attention_dropout, partition_size=partition_size, grid_size=self.grid_size, p_stochastic_dropout=p)]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        for layer in self.layers:
            x = layer(x)
        return x