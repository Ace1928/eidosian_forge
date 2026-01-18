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
class SetProduct_OrderedSet(_ScalarOrderedSetMixin, _OrderedSetMixin, SetProduct_FiniteSet):
    __slots__ = tuple()

    def at(self, index):
        _idx = self._to_0_based_index(index)
        _ord = list((len(_) for _ in self._sets))
        i = len(_ord)
        while i:
            i -= 1
            _ord[i], _idx = (_idx % _ord[i], _idx // _ord[i])
        if _idx:
            raise IndexError(f'{self.name} index out of range')
        ans = tuple((s.at(i + 1) for s, i in zip(self._sets, _ord)))
        if FLATTEN_CROSS_PRODUCT and normalize_index.flatten and (self.dimen != len(ans)):
            return self._flatten_product(ans)
        return ans

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        found = self._find_val(item)
        if found is None:
            raise IndexError('Cannot identify position of %s in Set %s: item not in Set' % (item, self.name))
        val, cutPoints = found
        if cutPoints is not None:
            val = tuple((val[cutPoints[i]:cutPoints[i + 1]] for i in range(len(self._sets))))
        _idx = tuple((s.ord(val[i]) - 1 for i, s in enumerate(self._sets)))
        _len = list((len(_) for _ in self._sets))
        _len.append(1)
        ans = 0
        for pos, n in zip(_idx, _len[1:]):
            ans += pos
            ans *= n
        return ans + 1