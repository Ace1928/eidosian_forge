import torch
import numbers
from torch.nn.parameter import Parameter
from .module import Module
from ._functions import CrossMapLRN2d as _cross_map_lrn2d
from .. import functional as F
from .. import init
from torch import Tensor, Size
from typing import Union, List, Tuple
class LocalResponseNorm(Module):
    """Applies local response normalization over an input signal.

    The input signal is composed of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\\left(k + \\frac{\\alpha}{n}
        \\sum_{c'=\\max(0, c-n/2)}^{\\min(N-1,c+n/2)}a_{c'}^2\\right)^{-\\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """
    __constants__ = ['size', 'alpha', 'beta', 'k']
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(self, size: int, alpha: float=0.0001, beta: float=0.75, k: float=1.0) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)