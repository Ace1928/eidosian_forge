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
class _FiniteSetMixin(object):
    __slots__ = ()

    def __len__(self):
        raise DeveloperError('Derived finite set class (%s) failed to implement __len__' % (type(self).__name__,))

    def _iter_impl(self):
        raise DeveloperError('Derived finite set class (%s) failed to implement _iter_impl' % (type(self).__name__,))

    def __iter__(self):
        """Iterate over the finite set

        Note: derived classes should NOT reimplement this method, and
        should instead overload _iter_impl.  The expression template
        system relies on being able to replace this method for all Sets
        during template generation.

        """
        return self._iter_impl()

    def __reversed__(self):
        return reversed(self.data())

    def sorted_iter(self):
        return iter(sorted_robust(self))

    def ordered_iter(self):
        return self.sorted_iter()

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return True

    def isfinite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return True

    def data(self):
        return tuple(self)

    @property
    @deprecated("The 'value' attribute is deprecated.  Use .data() to retrieve the values in a finite set.", version='5.7')
    def value(self):
        return set(self)

    @property
    @deprecated("The 'value_list' attribute is deprecated.  Use .ordered_data() to retrieve the values from a finite set in a deterministic order.", version='5.7')
    def value_list(self):
        return list(self.ordered_data())

    def sorted_data(self):
        return tuple(sorted_robust(self.data()))

    def ordered_data(self):
        return self.sorted_data()

    def bounds(self):
        try:
            lb = min(self, default=None)
            ub = max(self, default=None)
        except:
            lb = ub = None
        if type(lb) is not type(ub) and (type(lb) not in native_numeric_types or type(ub) not in native_numeric_types):
            return (None, None)
        else:
            return (lb, ub)

    def ranges(self):
        for i in self:
            if i.__class__ in native_numeric_types:
                yield NumericRange(i, i, 0)
            elif i.__class__ in native_types:
                yield NonNumericRange(i)
            else:
                try:
                    as_numeric(i)
                    yield NumericRange(i, i, 0)
                except:
                    yield NonNumericRange(i)