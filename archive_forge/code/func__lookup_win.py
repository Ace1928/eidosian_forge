import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _lookup_win(self, key, basename, styles, fail=False):
    for suffix in ('', ' (TrueType)'):
        for style in styles:
            try:
                valname = '%s%s%s' % (basename, style and ' ' + style, suffix)
                val, _ = _winreg.QueryValueEx(key, valname)
                return val
            except EnvironmentError:
                continue
    else:
        if fail:
            raise FontNotFound('Font %s (%s) not found in registry' % (basename, styles[0]))
        return None