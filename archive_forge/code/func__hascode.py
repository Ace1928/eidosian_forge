import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _hascode(object):
    """True if object has an attribute that stores it's __code__"""
    return getattr(object, '__code__', None) or getattr(object, 'func_code', None)