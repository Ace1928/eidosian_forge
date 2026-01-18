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
class _SpecialGenericAlias(_NotIterable, _BaseGenericAlias, _root=True):

    def __init__(self, origin, nparams, *, inst=True, name=None):
        if name is None:
            name = origin.__name__
        super().__init__(origin, inst=inst, name=name)
        self._nparams = nparams
        if origin.__module__ == 'builtins':
            self.__doc__ = f'A generic version of {origin.__qualname__}.'
        else:
            self.__doc__ = f'A generic version of {origin.__module__}.{origin.__qualname__}.'

    @_tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        msg = 'Parameters to generic types must be types.'
        params = tuple((_type_check(p, msg) for p in params))
        _check_generic(self, params, self._nparams)
        return self.copy_with(params)

    def copy_with(self, params):
        return _GenericAlias(self.__origin__, params, name=self._name, inst=self._inst)

    def __repr__(self):
        return 'typing.' + self._name

    def __subclasscheck__(self, cls):
        if isinstance(cls, _SpecialGenericAlias):
            return issubclass(cls.__origin__, self.__origin__)
        if not isinstance(cls, _GenericAlias):
            return issubclass(cls, self.__origin__)
        return super().__subclasscheck__(cls)

    def __reduce__(self):
        return self._name

    def __or__(self, right):
        return Union[self, right]

    def __ror__(self, left):
        return Union[left, self]