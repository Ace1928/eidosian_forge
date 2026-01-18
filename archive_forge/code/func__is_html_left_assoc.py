import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_html_left_assoc(k):
    return _is_html_assoc(k) or k == Z3_OP_SUB or k == Z3_OP_BSUB