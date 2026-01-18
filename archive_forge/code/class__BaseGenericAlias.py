from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
class _BaseGenericAlias(_Final, _root=True):
    """The central part of the internal API.

    This represents a generic version of type 'origin' with type arguments 'params'.
    There are two kind of these aliases: user defined and special. The special ones
    are wrappers around builtin collections and ABCs in collections.abc. These must
    have 'name' always set. If 'inst' is False, then the alias can't be instantiated;
    this is used by e.g. typing.List and typing.Dict.
    """

    def __init__(self, origin, *, inst=True, name=None):
        self._inst = inst
        self._name = name
        self.__origin__ = origin
        self.__slots__ = None

    def __call__(self, *args, **kwargs):
        if not self._inst:
            raise TypeError(f'Type {self._name} cannot be instantiated; use {self.__origin__.__name__}() instead')
        result = self.__origin__(*args, **kwargs)
        try:
            result.__orig_class__ = self
        except AttributeError:
            pass
        return result

    def __mro_entries__(self, bases):
        res = []
        if self.__origin__ not in bases:
            res.append(self.__origin__)
        i = bases.index(self)
        for b in bases[i + 1:]:
            if isinstance(b, _BaseGenericAlias) or issubclass(b, Generic):
                break
        else:
            res.append(Generic)
        return tuple(res)

    def __getattr__(self, attr):
        if attr in {'__name__', '__qualname__'}:
            return self._name or self.__origin__.__name__
        if '__origin__' in self.__dict__ and (not _is_dunder(attr)):
            return getattr(self.__origin__, attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if _is_dunder(attr) or attr in {'_name', '_inst', '_nparams', '_paramspec_tvars'}:
            super().__setattr__(attr, val)
        else:
            setattr(self.__origin__, attr, val)

    def __instancecheck__(self, obj):
        return self.__subclasscheck__(type(obj))

    def __subclasscheck__(self, cls):
        raise TypeError('Subscripted generics cannot be used with class and instance checks')

    def __dir__(self):
        return list(set(super().__dir__() + [attr for attr in dir(self.__origin__) if not _is_dunder(attr)]))