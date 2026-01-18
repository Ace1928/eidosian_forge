from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class AvgPool1d(_AvgPoolNd):
    """Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \\text{out}(N_i, C_j, l) = \\frac{1}{k} \\sum_{m=0}^{k-1}
                               \\text{input}(N_i, C_j, \\text{stride} \\times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \\left\\lfloor \\frac{L_{in} +
              2 \\times \\text{padding} - \\text{kernel\\_size}}{\\text{stride}} + 1\\right\\rfloor

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]]))
        tensor([[[2., 4., 6.]]])
    """
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_1_t, stride: _size_1_t=None, padding: _size_1_t=0, ceil_mode: bool=False, count_include_pad: bool=True) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)