import functools
import itertools
from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64
import pennylane as qml
from . import single_dispatch  # pylint:disable=unused-import
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like
def _compute_vn_entropy(density_matrix, base=None):
    """Compute the Von Neumann entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` tensor for an integer `N`.
        base (float, int): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        float: Von Neumann entropy of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_vn_entropy(x)
    0.6931472

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_vn_entropy(x, base=2)
    1.0

    """
    if base:
        div_base = np.log(base)
    else:
        div_base = 1
    evs = qml.math.eigvalsh(density_matrix)
    evs = qml.math.where(evs > 0, evs, 1.0)
    entropy = qml.math.entr(evs) / div_base
    return entropy