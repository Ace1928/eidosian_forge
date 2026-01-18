from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def _import_module(name):
    """Import module, returning the module after the last dot."""
    __import__(name)
    return sys.modules[name]