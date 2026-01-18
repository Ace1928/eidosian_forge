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
class SetIntersectInitializer(InitializerBase):
    """An Initializer that returns the intersection of two SetInitializers

    Users will typically not create a SetIntersectInitializer directly.
    Instead, SetInitializer.intersect() may return a SetInitializer that
    contains a SetIntersectInitializer instance.

    """
    __slots__ = ('_A', '_B')

    def __init__(self, setA, setB):
        self._A = setA
        self._B = setB

    def __call__(self, parent, idx):
        return SetIntersection(self._A(parent, idx), self._B(parent, idx))

    def constant(self):
        return self._A.constant() and self._B.constant()

    def contains_indices(self):
        return self._A.contains_indices() or self._B.contains_indices()

    def indices(self):
        if self._A.contains_indices():
            if self._B.contains_indices():
                if set(self._A.indices()) != set(self._B.indices()):
                    raise ValueError('SetIntersectInitializer contains two sub-initializers with inconsistent external indices')
            return self._A.indices()
        else:
            return self._B.indices()