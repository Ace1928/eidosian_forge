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
class _UtilityRegistrations:

    def __init__(self, utilities, utility_registrations):
        self._cache = defaultdict(_defaultdict_int)
        self._utilities = utilities
        self._utility_registrations = utility_registrations
        self.__populate_cache()

    def __populate_cache(self):
        for (p, _), data in iter(self._utility_registrations.items()):
            component = data[0]
            self.__cache_utility(p, component)

    def __cache_utility(self, provided, component):
        try:
            self._cache[provided][component] += 1
        except TypeError:
            prov = self._cache[provided] = _UnhashableComponentCounter(self._cache[provided])
            prov[component] += 1

    def __uncache_utility(self, provided, component):
        provided = self._cache[provided]
        count = provided[component]
        count -= 1
        if count == 0:
            del provided[component]
        else:
            provided[component] = count
        return count > 0

    def _is_utility_subscribed(self, provided, component):
        try:
            return self._cache[provided][component] > 0
        except TypeError:
            return False

    def registerUtility(self, provided, name, component, info, factory):
        subscribed = self._is_utility_subscribed(provided, component)
        self._utility_registrations[provided, name] = (component, info, factory)
        self._utilities.register((), provided, name, component)
        if not subscribed:
            self._utilities.subscribe((), provided, component)
        self.__cache_utility(provided, component)

    def unregisterUtility(self, provided, name, component):
        del self._utility_registrations[provided, name]
        self._utilities.unregister((), provided, name)
        subscribed = self.__uncache_utility(provided, component)
        if not subscribed:
            self._utilities.unsubscribe((), provided, component)