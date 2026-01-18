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
@implementer(IAdapterRegistration)
class AdapterRegistration:

    def __init__(self, registry, required, provided, name, component, doc):
        self.registry, self.required, self.provided, self.name, self.factory, self.info = (registry, required, provided, name, component, doc)

    def __repr__(self):
        return '{}({!r}, {}, {}, {!r}, {}, {!r})'.format(self.__class__.__name__, self.registry, '[' + ', '.join([r.__name__ for r in self.required]) + ']', getattr(self.provided, '__name__', None), self.name, getattr(self.factory, '__name__', repr(self.factory)), self.info)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __ne__(self, other):
        return repr(self) != repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __le__(self, other):
        return repr(self) <= repr(other)

    def __gt__(self, other):
        return repr(self) > repr(other)

    def __ge__(self, other):
        return repr(self) >= repr(other)