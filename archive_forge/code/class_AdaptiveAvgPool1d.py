from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\\text{output\\_size}`.

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)

    """
    output_size: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input, self.output_size)