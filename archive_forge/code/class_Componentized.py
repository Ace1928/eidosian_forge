from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
class Componentized:
    """I am a mixin to allow you to be adapted in various ways persistently.

    I define a list of persistent adapters.  This is to allow adapter classes
    to store system-specific state, and initialized on demand.  The
    getComponent method implements this.  You must also register adapters for
    this class for the interfaces that you wish to pass to getComponent.

    Many other classes and utilities listed here are present in Zope3; this one
    is specific to Twisted.
    """
    persistenceVersion = 1

    def __init__(self):
        self._adapterCache = {}

    def locateAdapterClass(self, klass, interfaceClass, default):
        return getAdapterFactory(klass, interfaceClass, default)

    def setAdapter(self, interfaceClass, adapterClass):
        """
        Cache a provider for the given interface, by adapting C{self} using
        the given adapter class.
        """
        self.setComponent(interfaceClass, adapterClass(self))

    def addAdapter(self, adapterClass, ignoreClass=0):
        """Utility method that calls addComponent.  I take an adapter class and
        instantiate it with myself as the first argument.

        @return: The adapter instantiated.
        """
        adapt = adapterClass(self)
        self.addComponent(adapt, ignoreClass)
        return adapt

    def setComponent(self, interfaceClass, component):
        """
        Cache a provider of the given interface.
        """
        self._adapterCache[reflect.qual(interfaceClass)] = component

    def addComponent(self, component, ignoreClass=0):
        """
        Add a component to me, for all appropriate interfaces.

        In order to determine which interfaces are appropriate, the component's
        provided interfaces will be scanned.

        If the argument 'ignoreClass' is True, then all interfaces are
        considered appropriate.

        Otherwise, an 'appropriate' interface is one for which its class has
        been registered as an adapter for my class according to the rules of
        getComponent.
        """
        for iface in declarations.providedBy(component):
            if ignoreClass or self.locateAdapterClass(self.__class__, iface, None) == component.__class__:
                self._adapterCache[reflect.qual(iface)] = component

    def unsetComponent(self, interfaceClass):
        """Remove my component specified by the given interface class."""
        del self._adapterCache[reflect.qual(interfaceClass)]

    def removeComponent(self, component):
        """
        Remove the given component from me entirely, for all interfaces for which
        it has been registered.

        @return: a list of the interfaces that were removed.
        """
        l = []
        for k, v in list(self._adapterCache.items()):
            if v is component:
                del self._adapterCache[k]
                l.append(reflect.namedObject(k))
        return l

    def getComponent(self, interface, default=None):
        """Create or retrieve an adapter for the given interface.

        If such an adapter has already been created, retrieve it from the cache
        that this instance keeps of all its adapters.  Adapters created through
        this mechanism may safely store system-specific state.

        If you want to register an adapter that will be created through
        getComponent, but you don't require (or don't want) your adapter to be
        cached and kept alive for the lifetime of this Componentized object,
        set the attribute 'temporaryAdapter' to True on your adapter class.

        If you want to automatically register an adapter for all appropriate
        interfaces (with addComponent), set the attribute 'multiComponent' to
        True on your adapter class.
        """
        k = reflect.qual(interface)
        if k in self._adapterCache:
            return self._adapterCache[k]
        else:
            adapter = interface.__adapt__(self)
            if adapter is not None and (not (hasattr(adapter, 'temporaryAdapter') and adapter.temporaryAdapter)):
                self._adapterCache[k] = adapter
                if hasattr(adapter, 'multiComponent') and adapter.multiComponent:
                    self.addComponent(adapter)
            if adapter is None:
                return default
            return adapter

    def __conform__(self, interface):
        return self.getComponent(interface)