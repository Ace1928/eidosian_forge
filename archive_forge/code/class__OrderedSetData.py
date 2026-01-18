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
class _OrderedSetData(_OrderedSetMixin, _FiniteSetData):
    """
    This class defines the base class for an ordered set of concrete data.

    In older Pyomo terms, this defines a "concrete" ordered set - that is,
    a set that "owns" the list of set members.  While this class actually
    implements a set ordered by insertion order, we make the "official"
    _InsertionOrderSetData an empty derivative class, so that

         issubclass(_SortedSetData, _InsertionOrderSetData) == False

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """
    __slots__ = ('_ordered_values',)

    def __init__(self, component):
        self._values = {}
        self._ordered_values = []
        _FiniteSetData.__init__(self, component=component)

    def _iter_impl(self):
        """
        Return an iterator for the set.
        """
        return iter(self._ordered_values)

    def __reversed__(self):
        return reversed(self._ordered_values)

    def _add_impl(self, value):
        self._values[value] = len(self._values)
        self._ordered_values.append(value)

    def remove(self, val):
        idx = self._values.pop(val)
        self._ordered_values.pop(idx)
        for i in range(idx, len(self._ordered_values)):
            self._values[self._ordered_values[i]] -= 1

    def discard(self, val):
        try:
            self.remove(val)
        except KeyError:
            pass

    def clear(self):
        self._values.clear()
        self._ordered_values = []

    def pop(self):
        try:
            ans = self.last()
        except IndexError:
            raise KeyError('pop from an empty set')
        self.discard(ans)
        return ans

    def at(self, index):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal _lookup and _values are (pythonically) 0-based.
        """
        i = self._to_0_based_index(index)
        try:
            return self._ordered_values[i]
        except IndexError:
            raise IndexError(f'{self.name} index out of range') from None

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        try:
            return self._values[item] + 1
        except KeyError:
            if item.__class__ is not tuple or len(item) > 1:
                raise ValueError('%s.ord(x): x not in %s' % (self.name, self.name))
        try:
            return self._values[item[0]] + 1
        except KeyError:
            raise ValueError('%s.ord(x): x not in %s' % (self.name, self.name))