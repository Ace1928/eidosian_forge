import sys
import threading
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug
class BackgroundJobFunc(BackgroundJobBase):
    """Run a function call as a background job (uses a separate thread)."""

    def __init__(self, func, *args, **kwargs):
        """Create a new job from a callable object.

        Any positional arguments and keyword args given to this constructor
        after the initial callable are passed directly to it."""
        if not callable(func):
            raise TypeError('first argument to BackgroundJobFunc must be callable')
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.strform = str(func)
        self._init()

    def call(self):
        return self.func(*self.args, **self.kwargs)