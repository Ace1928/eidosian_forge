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
def _set_qualname(cls, value):
    if isinstance(value, FunctionType):
        value.__qualname__ = f'{cls.__qualname__}.{value.__name__}'
    return value