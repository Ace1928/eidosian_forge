import numpy as np
from pennylane.fermi import FermiSentence, FermiWord
from .observable_hf import qubit_observable
def _spin2_matrix_elements(sz):
    """Builds the table of matrix elements
    :math:`\\langle \\bm{\\alpha}, \\bm{\\beta} \\vert \\hat{s}_1 \\cdot \\hat{s}_2 \\vert
    \\bm{\\gamma}, \\bm{\\delta} \\rangle` of the two-particle spin operator
    :math:`\\hat{s}_1 \\cdot \\hat{s}_2`.

    The matrix elements are evaluated using the expression

    .. math::

        \\langle ~ (\\alpha, s_{z_\\alpha});~ (\\beta, s_{z_\\beta}) ~ \\vert \\hat{s}_1 &&
        \\cdot \\hat{s}_2 \\vert ~ (\\gamma, s_{z_\\gamma}); ~ (\\delta, s_{z_\\gamma}) ~ \\rangle =
        \\delta_{\\alpha,\\delta} \\delta_{\\beta,\\gamma} \\\\
        && \\times \\left( \\frac{1}{2} \\delta_{s_{z_\\alpha}, s_{z_\\delta}+1}
        \\delta_{s_{z_\\beta}, s_{z_\\gamma}-1} + \\frac{1}{2} \\delta_{s_{z_\\alpha}, s_{z_\\delta}-1}
        \\delta_{s_{z_\\beta}, s_{z_\\gamma}+1} + s_{z_\\alpha} s_{z_\\beta}
        \\delta_{s_{z_\\alpha}, s_{z_\\delta}} \\delta_{s_{z_\\beta}, s_{z_\\gamma}} \\right),

    where :math:`\\alpha` and :math:`s_{z_\\alpha}` refer to the quantum numbers of the spatial
    function and the spin projection, respectively, of the single-particle state
    :math:`\\vert \\bm{\\alpha} \\rangle \\equiv \\vert \\alpha, s_{z_\\alpha} \\rangle`.

    Args:
        sz (array[float]): spin-projection of the single-particle states

    Returns:
        array: NumPy array with the table of matrix elements. The first four columns
        contain the indices :math:`\\bm{\\alpha}`, :math:`\\bm{\\beta}`, :math:`\\bm{\\gamma}`,
        :math:`\\bm{\\delta}` and the fifth column stores the computed matrix element.

    **Example**

    >>> sz = np.array([0.5, -0.5])
    >>> print(_spin2_matrix_elements(sz))
    [[ 0.    0.    0.    0.    0.25]
     [ 0.    1.    1.    0.   -0.25]
     [ 1.    0.    0.    1.   -0.25]
     [ 1.    1.    1.    1.    0.25]
     [ 0.    1.    0.    1.    0.5 ]
     [ 1.    0.    1.    0.    0.5 ]]
    """
    n = np.arange(sz.size)
    alpha = n.reshape(-1, 1, 1, 1)
    beta = n.reshape(1, -1, 1, 1)
    gamma = n.reshape(1, 1, -1, 1)
    delta = n.reshape(1, 1, 1, -1)
    mask = np.logical_and(alpha // 2 == delta // 2, beta // 2 == gamma // 2)
    diag_mask = np.logical_and(sz[alpha] == sz[delta], sz[beta] == sz[gamma])
    diag_indices = np.argwhere(np.logical_and(mask, diag_mask))
    diag_values = (sz[alpha] * sz[beta]).flatten()
    diag = np.vstack([diag_indices.T, diag_values]).T
    m1 = np.logical_and(sz[alpha] == sz[delta] + 1, sz[beta] == sz[gamma] - 1)
    m2 = np.logical_and(sz[alpha] == sz[delta] - 1, sz[beta] == sz[gamma] + 1)
    off_diag_mask = np.logical_and(mask, np.logical_or(m1, m2))
    off_diag_indices = np.argwhere(off_diag_mask)
    off_diag_values = np.full([len(off_diag_indices)], 0.5)
    off_diag = np.vstack([off_diag_indices.T, off_diag_values]).T
    return np.vstack([diag, off_diag])