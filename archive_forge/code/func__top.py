import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
@property
def _top(self):
    if self.stack:
        return self.stack[-1]
    else:
        return self