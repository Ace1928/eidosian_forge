from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
@implementer(IUnregistered)
class Unregistered(RegistrationEvent):
    """A component or factory was unregistered
    """