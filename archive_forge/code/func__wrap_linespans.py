from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_linespans(self, inner):
    s = self.linespans
    i = self.linenostart - 1
    for t, line in inner:
        if t:
            i += 1
            yield (1, '<span id="%s-%d">%s</span>' % (s, i, line))
        else:
            yield (0, line)