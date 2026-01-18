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
class _AnyMeta(type):

    def __instancecheck__(self, obj):
        if self is Any:
            raise TypeError('typing_extensions.Any cannot be used with isinstance()')
        return super().__instancecheck__(obj)

    def __repr__(self):
        if self is Any:
            return 'typing_extensions.Any'
        return super().__repr__()