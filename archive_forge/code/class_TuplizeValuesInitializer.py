import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
class TuplizeValuesInitializer(InitializerBase):
    """An initializer wrapper that will "tuplize" a sequence

    This initializer takes the result of another initializer, and if it
    is a sequence that does not already contain tuples, will convert it
    to a sequence of tuples, each of length 'dimen' before returning it.

    """
    __slots__ = ('_init', '_dimen')

    def __new__(cls, *args):
        if args == (None,):
            return None
        else:
            return super(TuplizeValuesInitializer, cls).__new__(cls)

    def __init__(self, _init):
        self._init = _init
        self._dimen = UnknownSetDimen

    def __call__(self, parent, index):
        _val = self._init(parent, index)
        if self._dimen in {1, None, UnknownSetDimen}:
            return _val
        elif _val is Set.Skip:
            return _val
        elif _val is None:
            return _val
        if not isinstance(_val, Sequence):
            _val = tuple(_val)
        if len(_val) == 0:
            return _val
        if isinstance(_val[0], tuple):
            return _val
        return self._tuplize(_val, parent, index)

    def constant(self):
        return self._init.constant()

    def contains_indices(self):
        return self._init.contains_indices()

    def indices(self):
        return self._init.indices()

    def _tuplize(self, _val, parent, index):
        d = self._dimen
        if len(_val) % d:
            raise TuplizeError('Cannot tuplize list data for set %%s%%s because its length %s is not a multiple of dimen=%s' % (len(_val), d))
        return list((tuple(_val[d * i:d * (i + 1)]) for i in range(len(_val) // d)))