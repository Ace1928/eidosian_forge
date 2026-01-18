import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def _check_arg_int(name, arg):
    if not isinstance(arg, int):
        raise TypeError('operation %s argument must be an int, got %s' % (name, type(arg).__name__))
    if not 0 <= arg <= 2147483647:
        raise ValueError('operation %s argument must be in the range 0..2,147,483,647' % name)