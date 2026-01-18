from numbers import Number
import math
import operator
import warnings
def _vectorOp(self, other, op):
    if isinstance(other, Vector):
        assert len(self) == len(other)
        return self.__class__((op(a, b) for a, b in zip(self, other)))
    if isinstance(other, Number):
        return self.__class__((op(v, other) for v in self))
    raise NotImplementedError()