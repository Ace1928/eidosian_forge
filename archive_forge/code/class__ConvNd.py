import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.ao.nn.intrinsic import _FusedModule
from typing import Tuple, TypeVar, Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
class _ConvNd(nn.modules.conv._ConvNd):
    _FLOAT_MODULE = MOD

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], dilation: Tuple[int, ...], transposed: bool, output_padding: Tuple[int, ...], groups: int, bias: bool, padding_mode: str, qconfig=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(cls, mod):
        """Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if issubclass(type(mod), _FusedModule):
            mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=mod.bias is not None, padding_mode=mod.padding_mode, qconfig=qconfig)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        """ This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None, self.padding_mode)
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        if issubclass(cls, _FusedModule):
            modules = [conv]
            assert hasattr(cls, '_FLOAT_RELU_MODULE')
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)
            fused = cls._FLOAT_MODULE(*modules)
            fused.train(self.training)
            return fused
        else:
            return conv