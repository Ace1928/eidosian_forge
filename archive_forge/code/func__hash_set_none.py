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
def _hash_set_none(cls, fields, globals):
    return None