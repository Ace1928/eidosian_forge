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
def _hash_add(cls, fields, globals):
    flds = [f for f in fields if (f.compare if f.hash is None else f.hash)]
    return _set_qualname(cls, _hash_fn(flds, globals))