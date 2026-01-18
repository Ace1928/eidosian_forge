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
def _dataclass_setstate(self, state):
    for field, value in zip(fields(self), state):
        object.__setattr__(self, field.name, value)