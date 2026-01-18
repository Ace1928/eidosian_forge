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
class SetSymmetricDifference(SetOperator):
    __slots__ = tuple()
    _operator = ' ^ '

    def __new__(cls, *args):
        if cls != SetSymmetricDifference:
            return super(SetSymmetricDifference, cls).__new__(cls)
        set0, set1 = SetOperator._checkArgs(*args)
        if set0[0] and set1[0]:
            cls = SetSymmetricDifference_OrderedSet
        elif set0[1] and set1[1]:
            cls = SetSymmetricDifference_FiniteSet
        else:
            cls = SetSymmetricDifference_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        assert len(self._sets) == 2
        for set_a, set_b in (self._sets, reversed(self._sets)):
            for a_r in set_a.ranges():
                yield from a_r.range_difference(set_b.ranges())

    @property
    def dimen(self):
        d0 = self._sets[0].dimen
        d1 = self._sets[1].dimen
        if d0 is None or d1 is None:
            return None
        if d0 is UnknownSetDimen or d1 is UnknownSetDimen:
            return UnknownSetDimen
        if d0 == d1:
            return d0
        else:
            return None