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
class _ConcatenateGenericAlias(list):
    __class__ = typing._GenericAlias
    _special = False

    def __init__(self, origin, args):
        super().__init__(args)
        self.__origin__ = origin
        self.__args__ = args

    def __repr__(self):
        _type_repr = typing._type_repr
        return f'{_type_repr(self.__origin__)}[{', '.join((_type_repr(arg) for arg in self.__args__))}]'

    def __hash__(self):
        return hash((self.__origin__, self.__args__))

    def __call__(self, *args, **kwargs):
        pass

    @property
    def __parameters__(self):
        return tuple((tp for tp in self.__args__ if isinstance(tp, (typing.TypeVar, ParamSpec))))