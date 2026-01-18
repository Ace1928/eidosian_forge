import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _pop_frame(self):
    self.nextcaller = self.pop()