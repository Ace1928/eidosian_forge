from __future__ import annotations
import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter
from .wsgi import ClosingIterator
class _ProxyLookup:
    """Descriptor that handles proxied attribute lookup for
    :class:`LocalProxy`.

    :param f: The built-in function this attribute is accessed through.
        Instead of looking up the special method, the function call
        is redone on the object.
    :param fallback: Return this function if the proxy is unbound
        instead of raising a :exc:`RuntimeError`.
    :param is_attr: This proxied name is an attribute, not a function.
        Call the fallback immediately to get the value.
    :param class_value: Value to return when accessed from the
        ``LocalProxy`` class directly. Used for ``__doc__`` so building
        docs still works.
    """
    __slots__ = ('bind_f', 'fallback', 'is_attr', 'class_value', 'name')

    def __init__(self, f: t.Callable | None=None, fallback: t.Callable | None=None, class_value: t.Any | None=None, is_attr: bool=False) -> None:
        bind_f: t.Callable[[LocalProxy, t.Any], t.Callable] | None
        if hasattr(f, '__get__'):

            def bind_f(instance: LocalProxy, obj: t.Any) -> t.Callable:
                return f.__get__(obj, type(obj))
        elif f is not None:

            def bind_f(instance: LocalProxy, obj: t.Any) -> t.Callable:
                return partial(f, obj)
        else:
            bind_f = None
        self.bind_f = bind_f
        self.fallback = fallback
        self.class_value = class_value
        self.is_attr = is_attr

    def __set_name__(self, owner: LocalProxy, name: str) -> None:
        self.name = name

    def __get__(self, instance: LocalProxy, owner: type | None=None) -> t.Any:
        if instance is None:
            if self.class_value is not None:
                return self.class_value
            return self
        try:
            obj = instance._get_current_object()
        except RuntimeError:
            if self.fallback is None:
                raise
            fallback = self.fallback.__get__(instance, owner)
            if self.is_attr:
                return fallback()
            return fallback
        if self.bind_f is not None:
            return self.bind_f(instance, obj)
        return getattr(obj, self.name)

    def __repr__(self) -> str:
        return f'proxy {self.name}'

    def __call__(self, instance: LocalProxy, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Support calling unbound methods from the class. For example,
        this happens with ``copy.copy``, which does
        ``type(x).__copy__(x)``. ``type(x)`` can't be proxied, so it
        returns the proxy type and descriptor.
        """
        return self.__get__(instance, type(instance))(*args, **kwargs)