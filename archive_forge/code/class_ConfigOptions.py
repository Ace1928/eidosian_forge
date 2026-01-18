import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
class ConfigOptions(object):
    OPTIONS = {}

    def __init__(self):
        self._values = self.OPTIONS.copy()

    def set(self, name, value=True):
        if name not in self.OPTIONS:
            raise NameError('Invalid flag: %s' % name)
        self._values[name] = value

    def unset(self, name):
        self.set(name, False)

    def _check_attr(self, name):
        if name not in self.OPTIONS:
            raise AttributeError('Invalid flag: %s' % name)

    def __getattr__(self, name):
        self._check_attr(name)
        return self._values[name]

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(ConfigOptions, self).__setattr__(name, value)
        else:
            self._check_attr(name)
            self._values[name] = value

    def __repr__(self):
        return 'Flags(%s)' % ', '.join(('%s=%s' % (k, v) for k, v in self._values.items() if v is not False))

    def copy(self):
        copy = type(self)()
        copy._values = self._values.copy()
        return copy

    def __eq__(self, other):
        return isinstance(other, ConfigOptions) and other._values == self._values

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(sorted(self._values.items())))