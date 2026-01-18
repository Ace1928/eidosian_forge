import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _deoptop(op):
    name = _all_opname[op]
    return _all_opmap[deoptmap[name]] if name in deoptmap else op