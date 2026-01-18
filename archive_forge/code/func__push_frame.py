import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _push_frame(self):
    frame = self.nextcaller or None
    self.append(frame)
    self.nextcaller = None
    return frame