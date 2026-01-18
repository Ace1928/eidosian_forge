import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
class _SpecialForm(typing._Final, _root=True):
    __slots__ = ('_name', '__doc__', '_getitem')

    def __init__(self, getitem):
        self._getitem = getitem
        self._name = getitem.__name__
        self.__doc__ = getitem.__doc__

    def __getattr__(self, item):
        if item in {'__name__', '__qualname__'}:
            return self._name
        raise AttributeError(item)

    def __mro_entries__(self, bases):
        raise TypeError(f'Cannot subclass {self!r}')

    def __repr__(self):
        return f'typing_extensions.{self._name}'

    def __reduce__(self):
        return self._name

    def __call__(self, *args, **kwds):
        raise TypeError(f'Cannot instantiate {self!r}')

    def __or__(self, other):
        return typing.Union[self, other]

    def __ror__(self, other):
        return typing.Union[other, self]

    def __instancecheck__(self, obj):
        raise TypeError(f'{self} cannot be used with isinstance()')

    def __subclasscheck__(self, cls):
        raise TypeError(f'{self} cannot be used with issubclass()')

    @typing._tp_cache
    def __getitem__(self, parameters):
        return self._getitem(self, parameters)