import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def _signature_bound_method(sig):
    """Private helper to transform signatures for unbound
    functions to bound methods.
    """
    params = tuple(sig.parameters.values())
    if not params or params[0].kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
        raise ValueError('invalid method signature')
    kind = params[0].kind
    if kind in (_POSITIONAL_OR_KEYWORD, _POSITIONAL_ONLY):
        params = params[1:]
    elif kind is not _VAR_POSITIONAL:
        raise ValueError('invalid argument type')
    return sig.replace(parameters=params)