import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class CountedCallGenerator(object):
    """Generator implementing the "counted call" initialization scheme

    This generator implements the older "counted call" scheme, where the
    first argument past the parent block is a monotonically-increasing
    integer beginning at `start_at`.
    """

    def __init__(self, ctype, fcn, scalar, parent, idx, start_at):
        self._count = start_at - 1
        if scalar:
            self._fcn = lambda c: self._filter(ctype, fcn(parent, c))
        elif idx.__class__ is tuple:
            self._fcn = lambda c: self._filter(ctype, fcn(parent, c, *idx))
        else:
            self._fcn = lambda c: self._filter(ctype, fcn(parent, c, idx))

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        return self._fcn(self._count)
    next = __next__

    @staticmethod
    def _filter(ctype, x):
        if x is None:
            raise ValueError('Counted %s rule returned None instead of %s.End.\n    Counted %s rules of the form fcn(model, count, *idx) will be called\n    repeatedly with an increasing count parameter until the rule returns\n    %s.End.  None is not a valid return value in this case due to the\n    likelihood that an error in the rule can incorrectly return None.' % ((ctype.__name__,) * 4))
        return x