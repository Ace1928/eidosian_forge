import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_nargs_pattern(self, action):
    nargs = action.nargs
    if nargs is None:
        nargs_pattern = '(-*A-*)'
    elif nargs == OPTIONAL:
        nargs_pattern = '(-*A?-*)'
    elif nargs == ZERO_OR_MORE:
        nargs_pattern = '(-*[A-]*)'
    elif nargs == ONE_OR_MORE:
        nargs_pattern = '(-*A[A-]*)'
    elif nargs == REMAINDER:
        nargs_pattern = '([-AO]*)'
    elif nargs == PARSER:
        nargs_pattern = '(-*A[-AO]*)'
    else:
        nargs_pattern = '(-*%s-*)' % '-*'.join('A' * nargs)
    if action.option_strings:
        nargs_pattern = nargs_pattern.replace('-*', '')
        nargs_pattern = nargs_pattern.replace('-', '')
    return nargs_pattern