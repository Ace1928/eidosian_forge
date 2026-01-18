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
def _cmp_fn(name, op, self_tuple, other_tuple, globals):
    return _create_fn(name, ('self', 'other'), ['if other.__class__ is self.__class__:', f' return {self_tuple}{op}{other_tuple}', 'return NotImplemented'], globals=globals)