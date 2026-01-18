import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
@implementer(IAdapterRegistry)
class AdapterRegistry(BaseAdapterRegistry):
    """
    A full implementation of ``IAdapterRegistry`` that adds support for
    sub-registries.
    """
    LookupClass = AdapterLookup

    def __init__(self, bases=()):
        self._v_subregistries = weakref.WeakKeyDictionary()
        super().__init__(bases)

    def _addSubregistry(self, r):
        self._v_subregistries[r] = 1

    def _removeSubregistry(self, r):
        if r in self._v_subregistries:
            del self._v_subregistries[r]

    def _setBases(self, bases):
        old = self.__dict__.get('__bases__', ())
        for r in old:
            if r not in bases:
                r._removeSubregistry(self)
        for r in bases:
            if r not in old:
                r._addSubregistry(self)
        super()._setBases(bases)

    def changed(self, originally_changed):
        super().changed(originally_changed)
        for sub in self._v_subregistries.keys():
            sub.changed(originally_changed)