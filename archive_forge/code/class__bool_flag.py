import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
class _bool_flag(object):

    def __init__(self, val):
        self._val = val

    def __bool__(self):
        return self._val

    def _op(self, *others):
        raise ValueError(f'{self._val!r} ({type(self._val).__name__}) is not a valid numeric type. Cannot compute bounds on expression.')

    def __repr__(self):
        return repr(self._val)
    __float__ = _op
    __int__ = _op
    __abs__ = _op
    __neg__ = _op
    __add__ = _op
    __sub__ = _op
    __mul__ = _op
    __div__ = _op
    __pow__ = _op
    __radd__ = _op
    __rsub__ = _op
    __rmul__ = _op
    __rdiv__ = _op
    __rpow__ = _op