import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
class Tanhshrink(Module):
    """Applies the element-wise function:

    .. math::
        \\text{Tanhshrink}(x) = x - \\tanh(x)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanhshrink.png

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.tanhshrink(input)