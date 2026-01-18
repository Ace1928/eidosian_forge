from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_pre(self, inner):
    style = []
    if self.prestyles:
        style.append(self.prestyles)
    if self.noclasses:
        style.append('line-height: 125%')
    style = '; '.join(style)
    if self.filename:
        yield (0, '<span class="filename">' + self.filename + '</span>')
    yield (0, '<pre' + (style and ' style="%s"' % style) + '><span></span>')
    for tup in inner:
        yield tup
    yield (0, '</pre>')