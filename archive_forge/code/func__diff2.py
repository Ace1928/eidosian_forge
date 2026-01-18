import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _diff2(i, j, ri, rj, alpha, beta):
    """Compute the second order differentiated integral needed for evaluating a kinetic integral.

    The second-order integral :math:`D_{ij}^2`, where :math:`i` and :math:`j` denote angular
    momentum components of Gaussian functions, is computed from overlap integrals :math:`S` and the
    Gaussian exponent :math:`\\beta` as
    [`Helgaker (1995) p804 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:

    .. math::

        D_{ij}^2 = j(j-1)S_{i,j-2}^0 - 2\\beta(2j+1)S_{i,j}^0 + 4\\beta^2 S_{i,j+2}^0.

    Args:
        i (integer): angular momentum component for the first Gaussian function
        j (integer): angular momentum component for the second Gaussian function
        ri (array[float]): position component of the first Gaussian function
        rj (array[float]): position component of the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function

    Returns:
        array[float]: second-order differentiated integral between two Gaussian functions
    """
    p = alpha + beta
    d1 = j * (j - 1) * qml.math.sqrt(np.pi / p) * expansion(i, j - 2, ri, rj, alpha, beta, 0)
    d2 = -2 * beta * (2 * j + 1) * qml.math.sqrt(np.pi / p) * expansion(i, j, ri, rj, alpha, beta, 0)
    d3 = 4 * beta ** 2 * qml.math.sqrt(np.pi / p) * expansion(i, j + 2, ri, rj, alpha, beta, 0)
    return d1 + d2 + d3