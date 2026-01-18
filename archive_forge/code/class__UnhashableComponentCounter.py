from collections import defaultdict
from zope.interface.adapter import AdapterRegistry
from zope.interface.declarations import implementedBy
from zope.interface.declarations import implementer
from zope.interface.declarations import implementer_only
from zope.interface.declarations import providedBy
from zope.interface.interface import Interface
from zope.interface.interfaces import ComponentLookupError
from zope.interface.interfaces import IAdapterRegistration
from zope.interface.interfaces import IComponents
from zope.interface.interfaces import IHandlerRegistration
from zope.interface.interfaces import ISpecification
from zope.interface.interfaces import ISubscriptionAdapterRegistration
from zope.interface.interfaces import IUtilityRegistration
from zope.interface.interfaces import Registered
from zope.interface.interfaces import Unregistered
class _UnhashableComponentCounter:

    def __init__(self, otherdict):
        self._data = [item for item in otherdict.items()]

    def __getitem__(self, key):
        for component, count in self._data:
            if component == key:
                return count
        return 0

    def __setitem__(self, component, count):
        for i, data in enumerate(self._data):
            if data[0] == component:
                self._data[i] = (component, count)
                return
        self._data.append((component, count))

    def __delitem__(self, component):
        for i, data in enumerate(self._data):
            if data[0] == component:
                del self._data[i]
                return
        raise KeyError(component)