import re
from . import lazy_regex
from .trace import mutter, warning
def _invalid_regex(repl):

    def _(m):
        warning("'%s' not allowed within a regular expression. Replacing with '%s'" % (m, repl))
        return repl
    return _