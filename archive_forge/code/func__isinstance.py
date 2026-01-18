import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _isinstance(object):
    """True if object is a class instance type (and is not a builtin)"""
    if _hascode(object) or isclass(object) or ismodule(object):
        return False
    if istraceback(object) or isframe(object) or iscode(object):
        return False
    if not getmodule(object) and getmodule(type(object)).__name__ in ['numpy']:
        return True
    _types = ('<class ', "<type 'instance'>")
    if not repr(type(object)).startswith(_types):
        return False
    if not getmodule(object) or object.__module__ in ['builtins', '__builtin__'] or getname(object, force=True) in ['array']:
        return False
    return True