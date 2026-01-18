import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _make_cached_property_getattr(cached_properties, original_getattr, cls):
    lines = ['def wrapper():', '    __class__ = _cls', '    def __getattr__(self, item, cached_properties=cached_properties, original_getattr=original_getattr, _cached_setattr_get=_cached_setattr_get):', '         func = cached_properties.get(item)', '         if func is not None:', '              result = func(self)', '              _setter = _cached_setattr_get(self)', '              _setter(item, result)', '              return result']
    if original_getattr is not None:
        lines.append('         return original_getattr(self, item)')
    else:
        lines.extend(["         if hasattr(super(), '__getattr__'):", '              return super().__getattr__(item)', '         original_error = f"\'{self.__class__.__name__}\' object has no attribute \'{item}\'"', '         raise AttributeError(original_error)'])
    lines.extend(['    return __getattr__', '__getattr__ = wrapper()'])
    unique_filename = _generate_unique_filename(cls, 'getattr')
    glob = {'cached_properties': cached_properties, '_cached_setattr_get': _obj_setattr.__get__, '_cls': cls, 'original_getattr': original_getattr}
    return _make_method('__getattr__', '\n'.join(lines), unique_filename, glob)