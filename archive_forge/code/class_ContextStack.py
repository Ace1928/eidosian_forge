import sys
import warnings
from collections import deque
from functools import wraps
class ContextStack(ExitStack):
    """Backwards compatibility alias for ExitStack"""

    def __init__(self):
        warnings.warn('ContextStack has been renamed to ExitStack', DeprecationWarning)
        super(ContextStack, self).__init__()

    def register_exit(self, callback):
        return self.push(callback)

    def register(self, callback, *args, **kwds):
        return self.callback(callback, *args, **kwds)

    def preserve(self):
        return self.pop_all()