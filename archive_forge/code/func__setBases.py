import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def _setBases(self, bases):
    old = self.__dict__.get('__bases__', ())
    for r in old:
        if r not in bases:
            r._removeSubregistry(self)
    for r in bases:
        if r not in old:
            r._addSubregistry(self)
    super()._setBases(bases)