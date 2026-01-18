import _string
import re as _re
from collections import ChainMap as _ChainMap
def _invalid(self, mo):
    i = mo.start('invalid')
    lines = self.template[:i].splitlines(keepends=True)
    if not lines:
        colno = 1
        lineno = 1
    else:
        colno = i - len(''.join(lines[:-1]))
        lineno = len(lines)
    raise ValueError('Invalid placeholder in string: line %d, col %d' % (lineno, colno))