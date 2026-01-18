import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from ..utils import _log_api_usage_once, _make_ntuple
class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]]=3, stride: Union[int, Tuple[int, int]]=1, padding: Optional[Union[int, Tuple[int, int], str]]=None, groups: int=1, norm_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.ReLU, dilation: Union[int, Tuple[int, int]]=1, inplace: Optional[bool]=True, bias: Optional[bool]=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, norm_layer, activation_layer, dilation, inplace, bias, torch.nn.Conv2d)