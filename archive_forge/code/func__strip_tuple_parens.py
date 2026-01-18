import time
import types
from ..trace import mutter
from ..transport import decorator
def _strip_tuple_parens(self, t):
    t = repr(t)
    if t[0] == '(' and t[-1] == ')':
        t = t[1:-1]
    return t