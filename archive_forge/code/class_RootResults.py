import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
class RootResults(OptimizeResult):
    """Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.
    method : str
        Root finding method used.

    """

    def __init__(self, root, iterations, function_calls, flag, method):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        if flag in flag_map:
            self.flag = flag_map[flag]
        else:
            self.flag = flag
        self.method = method