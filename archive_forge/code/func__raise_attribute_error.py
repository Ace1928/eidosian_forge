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
def _raise_attribute_error(self, name: str) -> Never:
    if name == '__name__':
        raise AttributeError('readonly attribute')
    elif name in {'__value__', '__type_params__', '__parameters__', '__module__'}:
        raise AttributeError(f"attribute '{name}' of 'typing.TypeAliasType' objects is not writable")
    else:
        raise AttributeError(f"'typing.TypeAliasType' object has no attribute '{name}'")