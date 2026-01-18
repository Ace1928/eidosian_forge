from pyomo.common import DeveloperError
from pyomo.common.collections import (
from pyomo.common.modeling import NOTSET
from pyomo.core.base.set import DeclareGlobalSet, Set, SetOf, OrderedSetOf, _SetDataBase
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.indexed_component import IndexedComponent, normalize_index
from pyomo.core.base.indexed_component_slice import (
from pyomo.core.base.util import flatten_tuple
from pyomo.common.deprecation import deprecated
class _ReferenceSet(collections_Set):
    """A set-like object whose values are defined by a slice.

    This implements a set-like object whose members are defined by a
    component slice (:py:class:`IndexedComponent_slice`).
    :py:class:`_ReferenceSet` differs from the
    :py:class:`_ReferenceDict` above in that it looks in the underlying
    component ``index_set()`` for values that match the slice, and not
    just the (sparse) indices defined by the slice.

    Parameters
    ----------
    component_slice : :py:class:`IndexedComponent_slice`
        The slice object that defines the "members" of this set

    """

    def __init__(self, component_slice):
        self._slice = component_slice

    def __contains__(self, key):
        try:
            next(self._get_iter(self._slice, key))
            return True
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            _iter = iter(self._slice)
            for item in _iter:
                if _iter.get_last_index_wildcards() == key:
                    return True
            return False
        except (StopIteration, LookupError):
            return False

    def __iter__(self):
        return self._slice.index_wildcard_keys(False)

    def __len__(self):
        return sum((1 for _ in self))

    def _get_iter(self, _slice, key):
        if key.__class__ not in (tuple, list):
            key = (key,)
        if normalize_index.flatten:
            key = flatten_tuple(key)
        return _IndexedComponent_slice_iter(_slice, _fill_in_known_wildcards(key, look_in_index=True), iter_over_index=True)

    def __str__(self):
        return 'ReferenceSet(%s)' % (self._slice,)

    def ordered_iter(self):
        return self._slice.index_wildcard_keys(SortComponents.ORDERED_INDICES)

    def sorted_iter(self):
        return self._slice.index_wildcard_keys(SortComponents.SORTED_INDICES)