from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class _IBaseAdapterRegistration(IRegistration):
    """Information about the registration of an adapter
    """
    factory = Attribute('The factory used to create adapters')
    required = Attribute('The adapted interfaces\n\n    This is a sequence of interfaces adapters by the registered\n    factory.  The factory will be caled with a sequence of objects, as\n    positional arguments, that provide these interfaces.\n    ')
    provided = Attribute('The interface provided by the adapters.\n\n    This interface is implemented by the factory\n    ')