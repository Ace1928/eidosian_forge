from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IRegistrationEvent(IObjectEvent):
    """An event that involves a registration"""