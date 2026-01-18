from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _get_css_classes(self, ttype):
    """Return the css classes of this token type prefixed with
        the classprefix option."""
    cls = self._get_css_class(ttype)
    while ttype not in STANDARD_TYPES:
        ttype = ttype.parent
        cls = self._get_css_class(ttype) + ' ' + cls
    return cls