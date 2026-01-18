import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
class _MulExpr(_BinaryExpr):
    _op = mul
    _op_str = '*'

    @property
    def trivially_zero(self):
        try:
            return self.args[0].trivially_zero or self.args[1].trivially_zero
        except Exception:
            return False