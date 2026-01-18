import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def _find_leaf(self, byorder, required, provided, name):
    required = tuple([_convert_None_to_Interface(r) for r in required])
    order = len(required)
    if len(byorder) <= order:
        return None
    components = byorder[order]
    key = required + (provided,)
    for k in key:
        d = components.get(k)
        if d is None:
            return None
        components = d
    return components.get(name)