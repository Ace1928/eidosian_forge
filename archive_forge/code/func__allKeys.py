import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
@classmethod
def _allKeys(cls, components, i, parent_k=()):
    if i == 0:
        for k, v in components.items():
            yield (parent_k + (k,), v)
    else:
        for k, v in components.items():
            new_parent_k = parent_k + (k,)
            yield from cls._allKeys(v, i - 1, new_parent_k)