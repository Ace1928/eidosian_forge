from inspect import signature
import numpy as np
from scipy.optimize import brute, shgo
import pennylane as qml
def _restrict_to_univariate(fn, arg_idx, par_idx, args, kwargs):
    """Restrict a function to a univariate function for given argument
    and parameter indices.

    Args:
        fn (callable): Multivariate function
        arg_idx (int): Index of the argument that contains the parameter to restrict
        par_idx (tuple[int]): Index of the parameter to restrict to within the argument
        args (tuple): Arguments at which to restrict the function.
        kwargs (dict): Keyword arguments at which to restrict the function.

    Returns:
        callable: Univariate restriction of ``fn``. That is, this callable takes
        a single float value as input and has the same return type as ``fn``.
        All arguments are set to the given ``args`` and the input value to this
        function is added to the marked parameter.
    """
    the_arg = args[arg_idx]
    if len(qml.math.shape(the_arg)) == 0:
        shift_vec = qml.math.ones_like(the_arg)
    else:
        shift_vec = qml.math.zeros_like(the_arg)
        shift_vec = qml.math.scatter_element_add(shift_vec, par_idx, 1.0)

    def _univariate_fn(x):
        return fn(*args[:arg_idx], the_arg + shift_vec * x, *args[arg_idx + 1:], **kwargs)
    return _univariate_fn