import types
import sys
import numbers
import functools
import copy
import inspect
def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r