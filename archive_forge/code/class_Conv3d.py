import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.ao.nn.intrinsic import _FusedModule
from typing import Tuple, TypeVar, Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
class Conv3d(_ConvNd, nn.Conv3d):
    """
    A Conv3d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv3d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv3d#torch.nn.Conv3d
    for documentation.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Conv3d
    _FLOAT_CONV_MODULE = nn.Conv3d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t=1, padding: Union[str, _size_3_t]=0, dilation: _size_3_t=1, groups: int=1, bias: bool=True, padding_mode: str='zeros', qconfig=None, device=None, dtype=None) -> None:
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(in_channels, out_channels, kernel_size_, stride=stride_, padding=padding_, dilation=dilation_, transposed=False, output_padding=_triple(0), groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig, device=device, dtype=dtype)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)