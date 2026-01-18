import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _get_const_value(op, arg, co_consts):
    """Helper to get the value of the const in a hasconst op.

       Returns the dereferenced constant if this is possible.
       Otherwise (if it is a LOAD_CONST and co_consts is not
       provided) returns the dis.UNKNOWN sentinel.
    """
    assert op in hasconst
    argval = UNKNOWN
    if op == LOAD_CONST:
        if co_consts is not None:
            argval = co_consts[arg]
    return argval