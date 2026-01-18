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
class FiniteSetOf(_FiniteSetMixin, SetOf):

    def get(self, value, default=None):
        if value.__class__ is tuple and len(value) == 1:
            if value[0] in self._ref:
                return value[0]
        if value in self._ref:
            return value
        return default

    def __len__(self):
        return len(self._ref)

    def _iter_impl(self):
        return iter(self._ref)

    def __reversed__(self):
        try:
            return reversed(self._ref)
        except:
            return reversed(self.data())