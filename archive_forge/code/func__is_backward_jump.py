import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _is_backward_jump(op):
    return 'JUMP_BACKWARD' in opname[op]