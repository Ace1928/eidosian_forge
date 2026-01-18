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
def _set_default(type_param, default):
    if isinstance(default, (tuple, list)):
        type_param.__default__ = tuple((typing._type_check(d, 'Default must be a type') for d in default))
    elif default != _marker:
        if isinstance(type_param, ParamSpec) and default is ...:
            type_param.__default__ = default
        else:
            type_param.__default__ = typing._type_check(default, 'Default must be a type')
    else:
        type_param.__default__ = None