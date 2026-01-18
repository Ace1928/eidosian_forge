import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _get_code_object(x):
    """Helper to handle methods, compiled or raw code objects, and strings."""
    if hasattr(x, '__func__'):
        x = x.__func__
    if hasattr(x, '__code__'):
        x = x.__code__
    elif hasattr(x, 'gi_code'):
        x = x.gi_code
    elif hasattr(x, 'ag_code'):
        x = x.ag_code
    elif hasattr(x, 'cr_code'):
        x = x.cr_code
    if isinstance(x, str):
        x = _try_compile(x, '<disassembly>')
    if hasattr(x, 'co_code'):
        return x
    raise TypeError("don't know how to disassemble %s objects" % type(x).__name__)