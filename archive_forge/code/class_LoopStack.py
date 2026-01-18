import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
class LoopStack:
    """a stack for LoopContexts that implements the context manager protocol
    to automatically pop off the top of the stack on context exit
    """

    def __init__(self):
        self.stack = []

    def _enter(self, iterable):
        self._push(iterable)
        return self._top

    def _exit(self):
        self._pop()
        return self._top

    @property
    def _top(self):
        if self.stack:
            return self.stack[-1]
        else:
            return self

    def _pop(self):
        return self.stack.pop()

    def _push(self, iterable):
        new = LoopContext(iterable)
        if self.stack:
            new.parent = self.stack[-1]
        return self.stack.append(new)

    def __getattr__(self, key):
        raise exceptions.RuntimeException('No loop context is established')

    def __iter__(self):
        return iter(self._top)