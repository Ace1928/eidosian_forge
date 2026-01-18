import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _compile_and_eval(script, globs, locs=None, filename=''):
    """
    "Exec" the script with the given global (globs) and local (locs) variables.
    """
    bytecode = compile(script, filename, 'exec')
    eval(bytecode, globs, locs)