import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def _safeReprVars(varsDictItems):
    """
    Convert a list of (name, object) pairs into (name, repr) pairs.

    L{twisted.python.reflect.safe_repr} is used to generate the repr, so no
    exceptions will be raised by faulty C{__repr__} methods.

    @param varsDictItems: a sequence of (name, value) pairs as returned by e.g.
        C{locals().items()}.
    @returns: a sequence of (name, repr) pairs.
    """
    return [(name, reflect.safe_repr(obj)) for name, obj in varsDictItems]