import re
from .parsing import get_parsing_context
from ..units import fold_constants
def _post_proc(self, arg):
    for pp in self._post_procs:
        arg = pp(arg)
    return arg