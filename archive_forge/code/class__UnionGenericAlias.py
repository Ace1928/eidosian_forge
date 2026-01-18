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
class _UnionGenericAlias(_NotIterable, _GenericAlias, _root=True):

    def copy_with(self, params):
        return Union[params]

    def __eq__(self, other):
        if not isinstance(other, (_UnionGenericAlias, types.UnionType)):
            return NotImplemented
        return set(self.__args__) == set(other.__args__)

    def __hash__(self):
        return hash(frozenset(self.__args__))

    def __repr__(self):
        args = self.__args__
        if len(args) == 2:
            if args[0] is type(None):
                return f'typing.Optional[{_type_repr(args[1])}]'
            elif args[1] is type(None):
                return f'typing.Optional[{_type_repr(args[0])}]'
        return super().__repr__()

    def __instancecheck__(self, obj):
        return self.__subclasscheck__(type(obj))

    def __subclasscheck__(self, cls):
        for arg in self.__args__:
            if issubclass(cls, arg):
                return True

    def __reduce__(self):
        func, (origin, args) = super().__reduce__()
        return (func, (Union, args))