import functools
from .autoray import (
from . import lazy
class CompilePython:
    """A simple compiler that unravels all autoray calls, optionally sharing
    intermediates and folding constants, converts this to a code object using
    ``compile``, then executes this using ``exec``.

    Parameters
    ----------
    fn : callable
        Function to compile - should have signature
        ``fn(*args, **kwargs) -> array``, with ``args`` and ``kwargs`` any
        nested combination of ``tuple``, ``list`` and ``dict`` objects
        containing arrays (or other constant arguments), and perform array
        operations on these using ``autoray.do``.
    fold_constants : bool, optional
        Whether to fold all constant array operations into the graph, which
        might increase memory usage.
    share_intermediates : bool, optional
        Whether to cache all computational nodes during the trace, so that any
        shared intermediate results can be identified.
    """

    def __init__(self, fn, fold_constants=True, share_intermediates=True):
        self._fn = fn
        self._fold_constants = fold_constants
        self._share_intermediates = share_intermediates
        self._jit_fn = None

    def setup(self, args, kwargs):
        """Convert the example arrays to lazy variables and trace them through
        the function.
        """
        variables = tree_map(lazy.array, (args, kwargs))
        if self._share_intermediates:
            with backend_like('autoray.lazy'), lazy.shared_intermediates():
                outs = self._fn(*variables[0], **variables[1])
        else:
            with backend_like('autoray.lazy'):
                outs = self._fn(*variables[0], **variables[1])
        return lazy.Function(variables, outs, fold_constants=self._fold_constants)

    def __call__(self, *args, array_backend=None, **kwargs):
        """If necessary, build, then call the compiled function."""
        if self._jit_fn is None:
            self._jit_fn = self.setup(args, kwargs)
        return self._jit_fn(args, kwargs)