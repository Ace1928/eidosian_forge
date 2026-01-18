import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
class UnaryWrapper(Expr):

    def __checks(self):
        if self.nargs != 1:
            raise ValueError('UnaryWrapper can only be used when nargs == 1')
        if self.unique_keys is not None:
            raise ValueError('UnaryWrapper can only be used when unique_keys are None')

    def __mul__(self, other):
        self.__checks()
        arg, = self.args
        return self.__class__([_MulExpr([arg, _implicit_conversion(other)])])

    def __truediv__(self, other):
        if other == 1:
            return self
        self.__checks()
        arg, = self.args
        return self.__class__([_DivExpr([arg, _implicit_conversion(other)])])

    def __rtruediv__(self, other):
        self.__checks()
        arg, = self.args
        return self.__class__([_DivExpr([_implicit_conversion(other), arg])])

    @classmethod
    def from_callback(cls, callback, attr='__call__', **kwargs):
        Wrapper = super().from_callback(callback, attr=attr, **kwargs)
        return lambda *args, **kw: cls(Wrapper(*args, **kw))