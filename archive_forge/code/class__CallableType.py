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
class _CallableType(_SpecialGenericAlias, _root=True):

    def copy_with(self, params):
        return _CallableGenericAlias(self.__origin__, params, name=self._name, inst=self._inst, _paramspec_tvars=True)

    def __getitem__(self, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError('Callable must be used as Callable[[arg, ...], result].')
        args, result = params
        if isinstance(args, list):
            params = (tuple(args), result)
        else:
            params = (args, result)
        return self.__getitem_inner__(params)

    @_tp_cache
    def __getitem_inner__(self, params):
        args, result = params
        msg = 'Callable[args, result]: result must be a type.'
        result = _type_check(result, msg)
        if args is Ellipsis:
            return self.copy_with((_TypingEllipsis, result))
        if not isinstance(args, tuple):
            args = (args,)
        args = tuple((_type_convert(arg) for arg in args))
        params = args + (result,)
        return self.copy_with(params)