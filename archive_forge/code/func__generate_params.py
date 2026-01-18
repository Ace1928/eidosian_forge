import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _generate_params(params, args):
    """Generate basis set parameters. The default values are used for the non-differentiable
    parameters and the user-defined values are used for the differentiable ones.

    Args:
        params (list(array[float])): default values of the basis set parameters
        args (list(array[float])): initial values of the differentiable basis set parameters

    Returns:
        list(array[float]): basis set parameters
    """
    basis_params = []
    c = 0
    for p in params:
        if p.requires_grad:
            basis_params.append(args[c])
            c += 1
        else:
            basis_params.append(p)
    return basis_params