import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from ..utils import _log_api_usage_once, _make_ntuple
class ConvNormActivation(torch.nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]]=3, stride: Union[int, Tuple[int, ...]]=1, padding: Optional[Union[int, Tuple[int, ...], str]]=None, groups: int=1, norm_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.ReLU, dilation: Union[int, Tuple[int, ...]]=1, inplace: Optional[bool]=True, bias: Optional[bool]=None, conv_layer: Callable[..., torch.nn.Module]=torch.nn.Conv2d) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple(((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)))
        if bias is None:
            bias = norm_layer is None
        layers = [conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {'inplace': inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels
        if self.__class__ == ConvNormActivation:
            warnings.warn("Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead.")