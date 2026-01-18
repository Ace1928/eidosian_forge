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
def _getUtilityProvided(component):
    provided = list(providedBy(component))
    if len(provided) == 1:
        return provided[0]
    raise TypeError("The utility doesn't provide a single interface and no provided interface was specified.")