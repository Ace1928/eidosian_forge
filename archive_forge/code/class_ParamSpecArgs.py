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
class ParamSpecArgs(_Immutable):
    """The args for a ParamSpec object.

        Given a ParamSpec object P, P.args is an instance of ParamSpecArgs.

        ParamSpecArgs objects have a reference back to their ParamSpec:

        P.args.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """

    def __init__(self, origin):
        self.__origin__ = origin

    def __repr__(self):
        return f'{self.__origin__.__name__}.args'

    def __eq__(self, other):
        if not isinstance(other, ParamSpecArgs):
            return NotImplemented
        return self.__origin__ == other.__origin__