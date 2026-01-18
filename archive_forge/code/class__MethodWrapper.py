from twisted.python import log
from twisted.words.xish import xpath
class _MethodWrapper:
    """
    Internal class for tracking method calls.
    """

    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        nargs = self.args + args
        nkwargs = self.kwargs.copy()
        nkwargs.update(kwargs)
        self.method(*nargs, **nkwargs)