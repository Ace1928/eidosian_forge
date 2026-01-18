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
def _repr_fn(fields, globals):
    fn = _create_fn('__repr__', ('self',), ['return self.__class__.__qualname__ + f"(' + ', '.join([f'{f.name}={{self.{f.name}!r}}' for f in fields]) + ')"'], globals=globals)
    return _recursive_repr(fn)