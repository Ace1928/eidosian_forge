import sys
import traceback
from mako import compat
from mako import util
def _format_filepos(lineno, pos, filename):
    if filename is None:
        return ' at line: %d char: %d' % (lineno, pos)
    else:
        return " in file '%s' at line: %d char: %d" % (filename, lineno, pos)