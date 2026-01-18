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
class _LiteralGenericAlias(typing._GenericAlias, _root=True):

    def __eq__(self, other):
        if not isinstance(other, _LiteralGenericAlias):
            return NotImplemented
        these_args_deduped = set(_value_and_type_iter(self.__args__))
        other_args_deduped = set(_value_and_type_iter(other.__args__))
        return these_args_deduped == other_args_deduped

    def __hash__(self):
        return hash(frozenset(_value_and_type_iter(self.__args__)))