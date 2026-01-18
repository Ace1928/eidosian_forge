import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
class Hardtanh(Module):
    """Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \\text{HardTanh}(x) = \\begin{cases}
            \\text{max\\_val} & \\text{ if } x > \\text{ max\\_val } \\\\
            \\text{min\\_val} & \\text{ if } x < \\text{ min\\_val } \\\\
            x & \\text{ otherwise } \\\\
        \\end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['min_val', 'max_val', 'inplace']
    min_val: float
    max_val: float
    inplace: bool

    def __init__(self, min_val: float=-1.0, max_val: float=1.0, inplace: bool=False, min_value: Optional[float]=None, max_value: Optional[float]=None) -> None:
        super().__init__()
        if min_value is not None:
            warnings.warn('keyword argument min_value is deprecated and rename to min_val')
            min_val = min_value
        if max_value is not None:
            warnings.warn('keyword argument max_value is deprecated and rename to max_val')
            max_val = max_value
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return f'min_val={self.min_val}, max_val={self.max_val}{inplace_str}'