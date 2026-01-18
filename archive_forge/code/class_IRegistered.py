from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IRegistered(IRegistrationEvent):
    """A component or factory was registered
    """