from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class MaxPool3d(_MaxPoolNd):
    """Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \\begin{aligned}
            \\text{out}(N_i, C_j, d, h, w) ={} & \\max_{k=0, \\ldots, kD-1} \\max_{m=0, \\ldots, kH-1} \\max_{n=0, \\ldots, kW-1} \\\\
                                              & \\text{input}(N_i, C_j, \\text{stride[0]} \\times d + k,
                                                             \\text{stride[1]} \\times h + m, \\text{stride[2]} \\times w + n)
        \\end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{dilation}[0] \\times
                (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

          .. math::
              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{dilation}[1] \\times
                (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

          .. math::
              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{dilation}[2] \\times
                (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: Tensor):
        return F.max_pool3d(input, self.kernel_size, self.stride, self.padding, self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)