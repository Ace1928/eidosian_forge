from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _copy_func_details(func, funcopy):
    funcopy.__name__ = func.__name__
    funcopy.__doc__ = func.__doc__
    try:
        funcopy.__text_signature__ = func.__text_signature__
    except AttributeError:
        pass
    try:
        funcopy.__module__ = func.__module__
    except AttributeError:
        pass
    try:
        funcopy.__defaults__ = func.__defaults__
    except AttributeError:
        pass
    try:
        funcopy.__kwdefaults__ = func.__kwdefaults__
    except AttributeError:
        pass
    if six.PY2:
        funcopy.func_defaults = func.func_defaults
        return