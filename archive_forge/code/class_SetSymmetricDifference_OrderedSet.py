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
class SetSymmetricDifference_OrderedSet(_ScalarOrderedSetMixin, _OrderedSetMixin, SetSymmetricDifference_FiniteSet):
    __slots__ = tuple()

    def at(self, index):
        idx = self._to_0_based_index(index)
        _iter = iter(self)
        try:
            while idx:
                next(_iter)
                idx -= 1
            return next(_iter)
        except StopIteration:
            raise IndexError(f'{self.name} index out of range') from None

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if item not in self:
            raise IndexError('Cannot identify position of %s in Set %s: item not in Set' % (item, self.name))
        idx = 0
        _iter = iter(self)
        while next(_iter) != item:
            idx += 1
        return idx + 1