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
class OrderedSetOf(_ScalarOrderedSetMixin, _OrderedSetMixin, FiniteSetOf):

    def at(self, index):
        i = self._to_0_based_index(index)
        try:
            return self._ref[i]
        except IndexError:
            raise IndexError(f'{self.name} index out of range') from None

    def ord(self, item):
        try:
            return self._ref.index(item) + 1
        except ValueError:
            if item.__class__ is not tuple or len(item) > 1:
                raise
        return self._ref.index(item[0]) + 1