import sys
import threading
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug
class BackgroundJobExpr(BackgroundJobBase):
    """Evaluate an expression as a background job (uses a separate thread)."""

    def __init__(self, expression, glob=None, loc=None):
        """Create a new job from a string which can be fed to eval().

        global/locals dicts can be provided, which will be passed to the eval
        call."""
        self.code = compile(expression, '<BackgroundJob compilation>', 'eval')
        glob = {} if glob is None else glob
        loc = {} if loc is None else loc
        self.expression = self.strform = expression
        self.glob = glob
        self.loc = loc
        self._init()

    def call(self):
        return eval(self.code, self.glob, self.loc)