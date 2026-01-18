from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IComponentRegistry(Interface):
    """Register components
    """

    def registerUtility(component=None, provided=None, name='', info='', factory=None):
        """Register a utility

        :param factory:
           Factory for the component to be registered.

        :param component:
           The registered component

        :param provided:
           This is the interface provided by the utility.  If the
           component provides a single interface, then this
           argument is optional and the component-implemented
           interface will be used.

        :param name:
           The utility name.

        :param info:
           An object that can be converted to a string to provide
           information about the registration.

        Only one of *component* and *factory* can be used.

        A `IRegistered` event is generated with an `IUtilityRegistration`.
        """

    def unregisterUtility(component=None, provided=None, name='', factory=None):
        """Unregister a utility

        :returns:
            A boolean is returned indicating whether the registry was
            changed.  If the given *component* is None and there is no
            component registered, or if the given *component* is not
            None and is not registered, then the function returns
            False, otherwise it returns True.

        :param factory:
           Factory for the component to be unregistered.

        :param component:
           The registered component The given component can be
           None, in which case any component registered to provide
           the given provided interface with the given name is
           unregistered.

        :param provided:
           This is the interface provided by the utility.  If the
           component is not None and provides a single interface,
           then this argument is optional and the
           component-implemented interface will be used.

        :param name:
           The utility name.

        Only one of *component* and *factory* can be used.
        An `IUnregistered` event is generated with an `IUtilityRegistration`.
        """

    def registeredUtilities():
        """Return an iterable of `IUtilityRegistration` instances.

        These registrations describe the current utility registrations
        in the object.
        """

    def registerAdapter(factory, required=None, provided=None, name='', info=''):
        """Register an adapter factory

        :param factory:
            The object used to compute the adapter

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set in class definitions using
            the `.adapter`
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param provided:
            This is the interface provided by the adapter and
            implemented by the factory.  If the factory
            implements a single interface, then this argument is
            optional and the factory-implemented interface will be
            used.

        :param name:
            The adapter name.

        :param info:
           An object that can be converted to a string to provide
           information about the registration.

        A `IRegistered` event is generated with an `IAdapterRegistration`.
        """

    def unregisterAdapter(factory=None, required=None, provided=None, name=''):
        """Unregister an adapter factory

        :returns:
            A boolean is returned indicating whether the registry was
            changed.  If the given component is None and there is no
            component registered, or if the given component is not
            None and is not registered, then the function returns
            False, otherwise it returns True.

        :param factory:
            This is the object used to compute the adapter. The
            factory can be None, in which case any factory
            registered to implement the given provided interface
            for the given required specifications with the given
            name is unregistered.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If the factory is not None and the required
            arguments is omitted, then the value of the factory's
            __component_adapts__ attribute will be used.  The
            __component_adapts__ attribute attribute is normally
            set in class definitions using adapts function, or for
            callables using the adapter decorator.  If the factory
            is None or doesn't have a __component_adapts__ adapts
            attribute, then this argument is required.

        :param provided:
            This is the interface provided by the adapter and
            implemented by the factory.  If the factory is not
            None and implements a single interface, then this
            argument is optional and the factory-implemented
            interface will be used.

        :param name:
            The adapter name.

        An `IUnregistered` event is generated with an `IAdapterRegistration`.
        """

    def registeredAdapters():
        """Return an iterable of `IAdapterRegistration` instances.

        These registrations describe the current adapter registrations
        in the object.
        """

    def registerSubscriptionAdapter(factory, required=None, provides=None, name='', info=''):
        """Register a subscriber factory

        :param factory:
            The object used to compute the adapter

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param provided:
            This is the interface provided by the adapter and
            implemented by the factory.  If the factory implements
            a single interface, then this argument is optional and
            the factory-implemented interface will be used.

        :param name:
            The adapter name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named subscribers is added.

        :param info:
           An object that can be converted to a string to provide
           information about the registration.

        A `IRegistered` event is generated with an
        `ISubscriptionAdapterRegistration`.
        """

    def unregisterSubscriptionAdapter(factory=None, required=None, provides=None, name=''):
        """Unregister a subscriber factory.

        :returns:
            A boolean is returned indicating whether the registry was
            changed.  If the given component is None and there is no
            component registered, or if the given component is not
            None and is not registered, then the function returns
            False, otherwise it returns True.

        :param factory:
            This is the object used to compute the adapter. The
            factory can be None, in which case any factories
            registered to implement the given provided interface
            for the given required specifications with the given
            name are unregistered.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param provided:
            This is the interface provided by the adapter and
            implemented by the factory.  If the factory is not
            None implements a single interface, then this argument
            is optional and the factory-implemented interface will
            be used.

        :param name:
            The adapter name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named subscribers is added.

        An `IUnregistered` event is generated with an
        `ISubscriptionAdapterRegistration`.
        """

    def registeredSubscriptionAdapters():
        """Return an iterable of `ISubscriptionAdapterRegistration` instances.

        These registrations describe the current subscription adapter
        registrations in the object.
        """

    def registerHandler(handler, required=None, name='', info=''):
        """Register a handler.

        A handler is a subscriber that doesn't compute an adapter
        but performs some function when called.

        :param handler:
            The object used to handle some event represented by
            the objects passed to it.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param name:
            The handler name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named handlers is added.

        :param info:
           An object that can be converted to a string to provide
           information about the registration.


        A `IRegistered` event is generated with an `IHandlerRegistration`.
        """

    def unregisterHandler(handler=None, required=None, name=''):
        """Unregister a handler.

        A handler is a subscriber that doesn't compute an adapter
        but performs some function when called.

        :returns: A boolean is returned indicating whether the registry was
            changed.

        :param handler:
            This is the object used to handle some event
            represented by the objects passed to it. The handler
            can be None, in which case any handlers registered for
            the given required specifications with the given are
            unregistered.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param name:
            The handler name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named handlers is added.

        An `IUnregistered` event is generated with an `IHandlerRegistration`.
        """

    def registeredHandlers():
        """Return an iterable of `IHandlerRegistration` instances.

        These registrations describe the current handler registrations
        in the object.
        """