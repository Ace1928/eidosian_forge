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
def _hash_exception(cls, fields, globals):
    raise TypeError(f'Cannot overwrite attribute __hash__ in class {cls.__name__}')