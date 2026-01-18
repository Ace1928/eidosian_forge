import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def _moment_matrix(*args):
    """Construct the multipole moment matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the multipole moment matrix
        """
    n = len(basis_functions)
    matrix = qml.math.zeros((n, n))
    for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
        args_ab = []
        if args:
            args_ab.extend(([arg[i], arg[j]] for arg in args))
        integral = moment_integral(a, b, order, idx, normalize=False)(*args_ab)
        o = qml.math.zeros((n, n))
        o[i, j] = o[j, i] = 1.0
        matrix = matrix + integral * o
    return matrix