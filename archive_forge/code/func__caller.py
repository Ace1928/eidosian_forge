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
def _caller(depth=2):
    try:
        return sys._getframe(depth).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        return None