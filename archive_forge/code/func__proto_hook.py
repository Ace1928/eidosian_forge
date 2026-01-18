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
@classmethod
def _proto_hook(cls, other):
    if not cls.__dict__.get('_is_protocol', False):
        return NotImplemented
    for attr in cls.__protocol_attrs__:
        for base in other.__mro__:
            if attr in base.__dict__:
                if base.__dict__[attr] is None:
                    return NotImplemented
                break
            annotations = getattr(base, '__annotations__', {})
            if isinstance(annotations, collections.abc.Mapping) and attr in annotations and is_protocol(other):
                break
        else:
            return NotImplemented
    return True