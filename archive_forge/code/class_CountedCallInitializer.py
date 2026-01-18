import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class CountedCallInitializer(InitializerBase):
    """Initializer for functions implementing the "counted call" API."""
    __slots__ = ('_fcn', '_is_counted_rule', '_scalar', '_ctype', '_start')

    def __init__(self, obj, _indexed_init, starting_index=1):
        self._fcn = _indexed_init._fcn
        self._is_counted_rule = None
        self._scalar = not obj.is_indexed()
        self._ctype = obj.ctype
        self._start = starting_index
        if self._scalar:
            self._is_counted_rule = True

    def __call__(self, parent, idx):
        if self._is_counted_rule == False:
            if idx.__class__ is tuple:
                return self._fcn(parent, *idx)
            else:
                return self._fcn(parent, idx)
        if self._is_counted_rule == True:
            return CountedCallGenerator(self._ctype, self._fcn, self._scalar, parent, idx, self._start)
        _args = inspect.getfullargspec(self._fcn)
        _nargs = len(_args.args)
        if inspect.ismethod(self._fcn) and self._fcn.__self__ is not None:
            _nargs -= 1
        _len = len(idx) if idx.__class__ is tuple else 1
        if _len + 2 == _nargs:
            self._is_counted_rule = True
        else:
            self._is_counted_rule = False
        return self.__call__(parent, idx)