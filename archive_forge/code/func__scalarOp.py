from numbers import Number
import math
import operator
import warnings
def _scalarOp(self, other, op):
    if isinstance(other, Number):
        return self.__class__((op(v, other) for v in self))
    raise NotImplementedError()