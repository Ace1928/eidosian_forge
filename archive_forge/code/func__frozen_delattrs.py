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
def _frozen_delattrs(self, name):
    """
    Attached to frozen classes as __delattr__.
    """
    raise FrozenInstanceError()