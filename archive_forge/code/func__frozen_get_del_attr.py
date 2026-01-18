import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _frozen_get_del_attr(cls, fields, globals):
    locals = {'cls': cls, 'FrozenInstanceError': FrozenInstanceError}
    if fields:
        fields_str = '(' + ','.join((repr(f.name) for f in fields)) + ',)'
    else:
        fields_str = '()'
    return (_create_fn('__setattr__', ('self', 'name', 'value'), (f'if type(self) is cls or name in {fields_str}:', ' raise FrozenInstanceError(f"cannot assign to field {name!r}")', f'super(cls, self).__setattr__(name, value)'), locals=locals, globals=globals), _create_fn('__delattr__', ('self', 'name'), (f'if type(self) is cls or name in {fields_str}:', ' raise FrozenInstanceError(f"cannot delete field {name!r}")', f'super(cls, self).__delattr__(name)'), locals=locals, globals=globals))