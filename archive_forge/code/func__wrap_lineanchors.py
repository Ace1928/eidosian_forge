from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_lineanchors(self, inner):
    s = self.lineanchors
    i = self.linenostart - 1
    for t, line in inner:
        if t:
            i += 1
            yield (1, '<a name="%s-%d"></a>' % (s, i) + line)
        else:
            yield (0, line)