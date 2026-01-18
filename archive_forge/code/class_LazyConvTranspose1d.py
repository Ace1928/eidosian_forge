import math
import warnings
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .lazy import LazyModuleMixin
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from ..common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):
    """A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose1d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = ConvTranspose1d

    def __init__(self, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t=1, padding: _size_1_t=0, output_padding: _size_1_t=0, groups: int=1, bias: bool=True, dilation: _size_1_t=1, padding_mode: str='zeros', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(0, 0, kernel_size, stride, padding, output_padding, groups, False, dilation, padding_mode, **factory_kwargs)
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 1