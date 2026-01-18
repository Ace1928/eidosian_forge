import re
from . import cursors, _mysql
from ._exceptions import (
def _tuple_literal(self, t):
    return b'(%s)' % b','.join(map(self.literal, t))