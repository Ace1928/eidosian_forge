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
def _is_classvar(a_type, typing):
    return a_type is typing.ClassVar or (type(a_type) is typing._GenericAlias and a_type.__origin__ is typing.ClassVar)