import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def __checks(self):
    if self.nargs != 1:
        raise ValueError('UnaryWrapper can only be used when nargs == 1')
    if self.unique_keys is not None:
        raise ValueError('UnaryWrapper can only be used when unique_keys are None')