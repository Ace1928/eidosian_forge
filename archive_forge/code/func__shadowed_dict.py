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
def _shadowed_dict(klass):
    dict_attr = type.__dict__['__dict__']
    for entry in _static_getmro(klass):
        try:
            class_dict = dict_attr.__get__(entry)['__dict__']
        except KeyError:
            pass
        else:
            if not (type(class_dict) is types.GetSetDescriptorType and class_dict.__name__ == '__dict__' and (class_dict.__objclass__ is entry)):
                return class_dict
    return _sentinel