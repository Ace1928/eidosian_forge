from .z3 import *
from .z3core import *
from .z3printer import *
def _to_rcfnum(num, ctx=None):
    if isinstance(num, RCFNum):
        return num
    else:
        return RCFNum(num, ctx)