from collections.abc import Iterable
import functools
import inspect
import numbers
import numpy as np
import pennylane as qml
def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (callable): a function

    Returns:
        dict[str, tuple]: mapping from argument name to (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {k: (idx, v.default) for idx, (k, v) in enumerate(signature.parameters.items()) if v.default is not inspect.Parameter.empty}