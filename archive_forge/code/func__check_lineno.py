import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def _check_lineno(lineno):
    if not isinstance(lineno, int):
        raise TypeError('lineno must be an int')
    if lineno < 1:
        raise ValueError('invalid lineno')