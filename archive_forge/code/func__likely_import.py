import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _likely_import(first, last, passive=False, explicit=True):
    return _getimport(first, last, verify=not passive, builtin=explicit)