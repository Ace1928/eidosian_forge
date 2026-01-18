import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def _set_module(typevarlike):
    def_mod = _caller(depth=3)
    if def_mod != 'typing_extensions':
        typevarlike.__module__ = def_mod