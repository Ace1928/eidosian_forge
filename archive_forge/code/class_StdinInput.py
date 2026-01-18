from __future__ import unicode_literals
from .utils import DummyContext, is_windows
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import io
import os
import sys
class StdinInput(Input):
    """
    Simple wrapper around stdin.
    """

    def __init__(self, stdin=None):
        self.stdin = stdin or sys.stdin
        assert self.stdin.isatty()
        try:
            self.stdin.fileno()
        except io.UnsupportedOperation:
            if 'idlelib.run' in sys.modules:
                raise io.UnsupportedOperation('Stdin is not a terminal. Running from Idle is not supported.')
            else:
                raise io.UnsupportedOperation('Stdin is not a terminal.')

    def __repr__(self):
        return 'StdinInput(stdin=%r)' % (self.stdin,)

    def raw_mode(self):
        return raw_mode(self.stdin.fileno())

    def cooked_mode(self):
        return cooked_mode(self.stdin.fileno())

    def fileno(self):
        return self.stdin.fileno()

    def read(self):
        return self.stdin.read()