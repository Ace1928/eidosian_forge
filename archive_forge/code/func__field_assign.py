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
def _field_assign(frozen, name, value, self_name):
    if frozen:
        return f'__dataclass_builtins_object__.__setattr__({self_name},{name!r},{value})'
    return f'{self_name}.{name}={value}'