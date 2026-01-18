import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def _aybabtu(c):
    """
    Get all of the parent classes of C{c}, not including C{c} itself, which are
    strict subclasses of L{Versioned}.

    @param c: a class
    @returns: list of classes
    """
    l = [c, Versioned]
    for b in inspect.getmro(c):
        if b not in l and issubclass(b, Versioned):
            l.append(b)
    return l[2:]