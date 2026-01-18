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
class _TypeVarLikeMeta(type):

    def __instancecheck__(cls, __instance: Any) -> bool:
        return isinstance(__instance, cls._backported_typevarlike)