import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
@_use_c_impl
class VerifyingBase(LookupBaseFallback):

    def changed(self, originally_changed):
        LookupBaseFallback.changed(self, originally_changed)
        self._verify_ro = self._registry.ro[1:]
        self._verify_generations = [r._generation for r in self._verify_ro]

    def _verify(self):
        if [r._generation for r in self._verify_ro] != self._verify_generations:
            self.changed(None)

    def _getcache(self, provided, name):
        self._verify()
        return LookupBaseFallback._getcache(self, provided, name)

    def lookupAll(self, required, provided):
        self._verify()
        return LookupBaseFallback.lookupAll(self, required, provided)

    def subscriptions(self, required, provided):
        self._verify()
        return LookupBaseFallback.subscriptions(self, required, provided)