from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
class IServiceCollection(Interface):
    """
    Collection of services.

    Contain several services, and manage their start-up/shut-down.
    Services can be accessed by name if they have a name, and it
    is always possible to iterate over them.
    """

    def getServiceNamed(name):
        """
        Get the child service with a given name.

        @type name: C{str}
        @rtype: L{IService}
        @raise KeyError: Raised if the service has no child with the
            given name.
        """

    def __iter__():
        """
        Get an iterator over all child services.
        """

    def addService(service):
        """
        Add a child service.

        Only implementations of L{IService.setServiceParent} should use this
        method.

        @type service: L{IService}
        @raise RuntimeError: Raised if the service has a child with
            the given name.
        """

    def removeService(service):
        """
        Remove a child service.

        Only implementations of L{IService.disownServiceParent} should
        use this method.

        @type service: L{IService}
        @raise ValueError: Raised if the given service is not a child.
        @rtype: L{Deferred<defer.Deferred>}
        @return: a L{Deferred<defer.Deferred>} which is triggered when the
            service has finished shutting down. If shutting down is immediate,
            a value can be returned (usually, L{None}).
        """