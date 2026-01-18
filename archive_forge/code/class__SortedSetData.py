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
class _SortedSetData(_SortedSetMixin, _OrderedSetData):
    """
    This class defines the data for a sorted set.

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """
    __slots__ = ('_is_sorted',)

    def __init__(self, component):
        self._is_sorted = True
        _OrderedSetData.__init__(self, component=component)

    def _iter_impl(self):
        """
        Return an iterator for the set.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self)._iter_impl()

    def __reversed__(self):
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).__reversed__()

    def _add_impl(self, value):
        self._values[value] = len(self._values)
        self._ordered_values.append(value)
        self._is_sorted = False

    def clear(self):
        super(_SortedSetData, self).clear()
        self._is_sorted = True

    def at(self, index):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal _lookup and _values are (pythonically) 0-based.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).at(index)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).ord(item)

    def sorted_data(self):
        return self.data()

    def _sort(self):
        self._ordered_values = list(self.parent_component()._sort_fcn(self._ordered_values))
        self._values = {j: i for i, j in enumerate(self._ordered_values)}
        self._is_sorted = True