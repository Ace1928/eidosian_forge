import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
class CallerStack(list):

    def __init__(self):
        self.nextcaller = None

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        return len(self) and self._get_caller() and True or False

    def _get_caller(self):
        return self[-1]

    def __getattr__(self, key):
        return getattr(self._get_caller(), key)

    def _push_frame(self):
        frame = self.nextcaller or None
        self.append(frame)
        self.nextcaller = None
        return frame

    def _pop_frame(self):
        self.nextcaller = self.pop()