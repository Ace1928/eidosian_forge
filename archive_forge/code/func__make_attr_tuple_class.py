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
def _make_attr_tuple_class(cls_name, attr_names):
    """
    Create a tuple subclass to hold `Attribute`s for an `attrs` class.

    The subclass is a bare tuple with properties for names.

    class MyClassAttributes(tuple):
        __slots__ = ()
        x = property(itemgetter(0))
    """
    attr_class_name = f'{cls_name}Attributes'
    attr_class_template = [f'class {attr_class_name}(tuple):', '    __slots__ = ()']
    if attr_names:
        for i, attr_name in enumerate(attr_names):
            attr_class_template.append(f'    {attr_name} = _attrs_property(_attrs_itemgetter({i}))')
    else:
        attr_class_template.append('    pass')
    globs = {'_attrs_itemgetter': itemgetter, '_attrs_property': property}
    _compile_and_eval('\n'.join(attr_class_template), globs)
    return globs[attr_class_name]