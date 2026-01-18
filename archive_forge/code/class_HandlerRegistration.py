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
@implementer_only(IHandlerRegistration)
class HandlerRegistration(AdapterRegistration):

    def __init__(self, registry, required, name, handler, doc):
        self.registry, self.required, self.name, self.handler, self.info = (registry, required, name, handler, doc)

    @property
    def factory(self):
        return self.handler
    provided = None

    def __repr__(self):
        return '{}({!r}, {}, {!r}, {}, {!r})'.format(self.__class__.__name__, self.registry, '[' + ', '.join([r.__name__ for r in self.required]) + ']', self.name, getattr(self.factory, '__name__', repr(self.factory)), self.info)