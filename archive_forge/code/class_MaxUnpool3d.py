from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class MaxUnpool3d(_MaxUnpoolNd):
    """Computes a partial inverse of :class:`MaxPool3d`.

    :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
    :class:`MaxUnpool3d` takes in as input the output of :class:`MaxPool3d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool3d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs section below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool3d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = (D_{in} - 1) \\times \\text{stride[0]} - 2 \\times \\text{padding[0]} + \\text{kernel\\_size[0]}

          .. math::
              H_{out} = (H_{in} - 1) \\times \\text{stride[1]} - 2 \\times \\text{padding[1]} + \\text{kernel\\_size[1]}

          .. math::
              W_{out} = (W_{in} - 1) \\times \\text{stride[2]} - 2 \\times \\text{padding[2]} + \\text{kernel\\_size[2]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # pool of square window of size=3, stride=2
        >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, stride=2)
        >>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
    """
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t

    def __init__(self, kernel_size: _size_3_t, stride: Optional[_size_3_t]=None, padding: _size_3_t=0) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding = _triple(padding)

    def forward(self, input: Tensor, indices: Tensor, output_size: Optional[List[int]]=None) -> Tensor:
        return F.max_unpool3d(input, indices, self.kernel_size, self.stride, self.padding, output_size)