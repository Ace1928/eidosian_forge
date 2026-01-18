import re
from . import lazy_regex
from .trace import mutter, warning
def _do_sub(self, m):
    fun = self._funs[m.lastindex - 1]
    if hasattr(fun, '__call__'):
        return fun(m.group(0))
    else:
        return self._expand.sub(m.group(0), fun)