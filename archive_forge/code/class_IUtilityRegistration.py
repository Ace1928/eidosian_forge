from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IUtilityRegistration(IRegistration):
    """Information about the registration of a utility
    """
    factory = Attribute('The factory used to create the utility. Optional.')
    component = Attribute('The object registered')
    provided = Attribute('The interface provided by the component')