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
def _get_protocol_attrs(cls):
    attrs = set()
    for base in cls.__mro__[:-1]:
        if base.__name__ in {'Protocol', 'Generic'}:
            continue
        annotations = getattr(base, '__annotations__', {})
        for attr in (*base.__dict__, *annotations):
            if not attr.startswith('_abc_') and attr not in _EXCLUDED_ATTRS:
                attrs.add(attr)
    return attrs