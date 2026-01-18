import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_html_infix(k):
    global _html_infix_map
    return _html_infix_map.get(k, False)