import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _push_buffer(self):
    """push a capturing buffer onto this Context."""
    self._push_writer()