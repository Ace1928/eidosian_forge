import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
class _BinaryExpr(Expr):
    _op = None

    def _str(self, *args, **kwargs):
        return ('({0} %s {1})' % self._op_str).format(*[arg._str(*args, **kwargs) for arg in self.args])

    def __repr__(self):
        return super(_BinaryExpr, self)._str(repr)

    def __call__(self, variables, backend=math, **kwargs):
        arg0, arg1 = self.all_args(variables, backend=backend, **kwargs)
        return self._op(arg0, arg1)

    def rate_coeff(self, *args, **kwargs):
        return self._op(self.args[0].rate_coeff(*args, **kwargs), self.args[1].rate_coeff(*args, **kwargs))