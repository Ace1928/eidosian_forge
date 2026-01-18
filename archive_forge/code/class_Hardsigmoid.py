import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
class Hardsigmoid(Module):
    """Applies the Hardsigmoid function element-wise.

    Hardsigmoid is defined as:

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\
            1 & \\text{if~} x \\ge +3, \\\\
            x / 6 + 1 / 2 & \\text{otherwise}
        \\end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardsigmoid.png

    Examples::

        >>> m = nn.Hardsigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool=False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, self.inplace)