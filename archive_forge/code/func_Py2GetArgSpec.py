from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def Py2GetArgSpec(fn):
    """A wrapper around getargspec that tries both fn and fn.__call__."""
    try:
        return inspect.getargspec(fn)
    except TypeError:
        if hasattr(fn, '__call__'):
            return inspect.getargspec(fn.__call__)
        raise