import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
class RefType(Enum):
    """
    Enumerate the reference type
    """
    '\n    A new reference\n    '
    NEW = 1
    '\n    A borrowed reference\n    '
    BORROWED = 2
    '\n    An untracked reference\n    '
    UNTRACKED = 3