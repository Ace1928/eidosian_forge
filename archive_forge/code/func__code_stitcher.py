import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _code_stitcher(block):
    """stitch together the strings in tuple 'block'"""
    if block[0] and block[-1]:
        block = '\n'.join(block)
    elif block[0]:
        block = block[0]
    elif block[-1]:
        block = block[-1]
    else:
        block = ''
    return block