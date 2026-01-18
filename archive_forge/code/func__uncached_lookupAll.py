import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def _uncached_lookupAll(self, required, provided):
    required = tuple(required)
    order = len(required)
    result = {}
    for registry in reversed(self._registry.ro):
        byorder = registry._adapters
        if order >= len(byorder):
            continue
        extendors = registry._v_lookup._extendors.get(provided)
        if not extendors:
            continue
        components = byorder[order]
        _lookupAll(components, required, extendors, result, 0, order)
    self._subscribe(*required)
    return tuple(result.items())