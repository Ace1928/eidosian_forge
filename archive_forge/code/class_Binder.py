import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
class Binder:
    """Bind interfaces to implementations.

    .. note:: This class is instantiated internally for you and there's no need
        to instantiate it on your own.
    """
    _bindings: Dict[type, Binding]

    @private
    def __init__(self, injector: 'Injector', auto_bind: bool=True, parent: Optional['Binder']=None) -> None:
        """Create a new Binder.

        :param injector: Injector we are binding for.
        :param auto_bind: Whether to automatically bind missing types.
        :param parent: Parent binder.
        """
        self.injector = injector
        self._auto_bind = auto_bind
        self._bindings = {}
        self.parent = parent

    def bind(self, interface: Type[T], to: Union[None, T, Callable[..., T], Provider[T]]=None, scope: Union[None, Type['Scope'], 'ScopeDecorator']=None) -> None:
        """Bind an interface to an implementation.

        Binding `T` to an instance of `T` like

        ::

            binder.bind(A, to=A('some', 'thing'))

        is, for convenience, a shortcut for

        ::

            binder.bind(A, to=InstanceProvider(A('some', 'thing'))).

        Likewise, binding to a callable like

        ::

            binder.bind(A, to=some_callable)

        is a shortcut for

        ::

            binder.bind(A, to=CallableProvider(some_callable))

        and, as such, if `some_callable` there has any annotated parameters they'll be provided
        automatically without having to use :func:`inject` or :data:`Inject` with the callable.

        `typing.List` and `typing.Dict` instances are reserved for multibindings and trying to bind them
        here will result in an error (use :meth:`multibind` instead)::

            binder.bind(List[str], to=['hello', 'there'])  # Error

        :param interface: Type to bind.
        :param to: Instance or class to bind to, or an instance of
             :class:`Provider` subclass.
        :param scope: Optional :class:`Scope` in which to bind.
        """
        if _get_origin(_punch_through_alias(interface)) in {dict, list}:
            raise Error('Type %s is reserved for multibindings. Use multibind instead of bind.' % (interface,))
        self._bindings[interface] = self.create_binding(interface, to, scope)

    @overload
    def multibind(self, interface: Type[List[T]], to: Union[List[T], Callable[..., List[T]], Provider[List[T]]], scope: Union[Type['Scope'], 'ScopeDecorator', None]=None) -> None:
        pass

    @overload
    def multibind(self, interface: Type[Dict[K, V]], to: Union[Dict[K, V], Callable[..., Dict[K, V]], Provider[Dict[K, V]]], scope: Union[Type['Scope'], 'ScopeDecorator', None]=None) -> None:
        pass

    def multibind(self, interface: type, to: Any, scope: Union['ScopeDecorator', Type['Scope'], None]=None) -> None:
        """Creates or extends a multi-binding.

        A multi-binding contributes values to a list or to a dictionary. For example::

            binder.multibind(List[str], to=['some', 'strings'])
            binder.multibind(List[str], to=['other', 'strings'])
            injector.get(List[str])  # ['some', 'strings', 'other', 'strings']

            binder.multibind(Dict[str, int], to={'key': 11})
            binder.multibind(Dict[str, int], to={'other_key': 33})
            injector.get(Dict[str, int])  # {'key': 11, 'other_key': 33}

        .. versionchanged:: 0.17.0
            Added support for using `typing.Dict` and `typing.List` instances as interfaces.
            Deprecated support for `MappingKey`, `SequenceKey` and single-item lists and
            dictionaries as interfaces.

        :param interface: typing.Dict or typing.List instance to bind to.
        :param to: Instance, class to bind to, or an explicit :class:`Provider`
                subclass. Must provide a list or a dictionary, depending on the interface.
        :param scope: Optional Scope in which to bind.
        """
        if interface not in self._bindings:
            provider: ListOfProviders
            if isinstance(interface, dict) or (isinstance(interface, type) and issubclass(interface, dict)) or _get_origin(_punch_through_alias(interface)) is dict:
                provider = MapBindProvider()
            else:
                provider = MultiBindProvider()
            binding = self.create_binding(interface, provider, scope)
            self._bindings[interface] = binding
        else:
            binding = self._bindings[interface]
            provider = binding.provider
            assert isinstance(provider, ListOfProviders)
        provider.append(self.provider_for(interface, to))

    def install(self, module: _InstallableModuleType) -> None:
        """Install a module into this binder.

        In this context the module is one of the following:

        * function taking the :class:`Binder` as it's only parameter

          ::

            def configure(binder):
                bind(str, to='s')

            binder.install(configure)

        * instance of :class:`Module` (instance of it's subclass counts)

          ::

            class MyModule(Module):
                def configure(self, binder):
                    binder.bind(str, to='s')

            binder.install(MyModule())

        * subclass of :class:`Module` - the subclass needs to be instantiable so if it
          expects any parameters they need to be injected

          ::

            binder.install(MyModule)
        """
        if type(module) is type and issubclass(cast(type, module), Module):
            instance = cast(type, module)()
        else:
            instance = module
        instance(self)

    def create_binding(self, interface: type, to: Any=None, scope: Union['ScopeDecorator', Type['Scope'], None]=None) -> Binding:
        provider = self.provider_for(interface, to)
        scope = scope or getattr(to or interface, '__scope__', NoScope)
        if isinstance(scope, ScopeDecorator):
            scope = scope.scope
        return Binding(interface, provider, scope)

    def provider_for(self, interface: Any, to: Any=None) -> Provider:
        base_type = _punch_through_alias(interface)
        origin = _get_origin(base_type)
        if interface is Any:
            raise TypeError('Injecting Any is not supported')
        elif _is_specialization(interface, ProviderOf):
            target, = interface.__args__
            if to is not None:
                raise Exception('ProviderOf cannot be bound to anything')
            return InstanceProvider(ProviderOf(self.injector, target))
        elif isinstance(to, Provider):
            return to
        elif isinstance(to, (types.FunctionType, types.LambdaType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
            return CallableProvider(to)
        elif issubclass(type(to), type):
            return ClassProvider(cast(type, to))
        elif isinstance(interface, BoundKey):

            def proxy(injector: Injector) -> Any:
                binder = injector.binder
                kwarg_providers = {name: binder.provider_for(None, provider) for name, provider in interface.kwargs.items()}
                kwargs = {name: provider.get(injector) for name, provider in kwarg_providers.items()}
                return interface.interface(**kwargs)
            return CallableProvider(inject(proxy))
        elif _is_specialization(interface, AssistedBuilder):
            target, = interface.__args__
            builder = interface(self.injector, target)
            return InstanceProvider(builder)
        elif origin is None and isinstance(base_type, (tuple, type)) and (interface is not Any) and isinstance(to, base_type) or (origin in {dict, list} and isinstance(to, origin)):
            return InstanceProvider(to)
        elif issubclass(type(base_type), type) or isinstance(base_type, (tuple, list)):
            if to is not None:
                return InstanceProvider(to)
            return ClassProvider(base_type)
        else:
            raise UnknownProvider("couldn't determine provider for %r to %r" % (interface, to))

    def _get_binding(self, key: type, *, only_this_binder: bool=False) -> Tuple[Binding, 'Binder']:
        binding = self._bindings.get(key)
        if binding:
            return (binding, self)
        if self.parent and (not only_this_binder):
            return self.parent._get_binding(key)
        raise KeyError

    def get_binding(self, interface: type) -> Tuple[Binding, 'Binder']:
        is_scope = isinstance(interface, type) and issubclass(interface, Scope)
        is_assisted_builder = _is_specialization(interface, AssistedBuilder)
        try:
            return self._get_binding(interface, only_this_binder=is_scope or is_assisted_builder)
        except (KeyError, UnsatisfiedRequirement):
            if is_scope:
                scope = interface
                self.bind(scope, to=scope(self.injector))
                return self._get_binding(interface)
            if self._auto_bind or self._is_special_interface(interface):
                binding = ImplicitBinding(*self.create_binding(interface))
                self._bindings[interface] = binding
                return (binding, self)
        raise UnsatisfiedRequirement(None, interface)

    def has_binding_for(self, interface: type) -> bool:
        return interface in self._bindings

    def has_explicit_binding_for(self, interface: type) -> bool:
        return self.has_binding_for(interface) and (not isinstance(self._bindings[interface], ImplicitBinding))

    def _is_special_interface(self, interface: type) -> bool:
        return any((_is_specialization(interface, cls) for cls in [AssistedBuilder, ProviderOf]))