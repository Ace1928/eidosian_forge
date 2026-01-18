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
def _compute_mutual_info(state, indices0, indices1, base=None, check_state=False, c_dtype='complex128'):
    """Compute the mutual information between the subsystems."""
    all_indices = sorted([*indices0, *indices1])
    vn_entropy_1 = vn_entropy(state, indices=indices0, base=base, check_state=check_state, c_dtype=c_dtype)
    vn_entropy_2 = vn_entropy(state, indices=indices1, base=base, check_state=check_state, c_dtype=c_dtype)
    vn_entropy_12 = vn_entropy(state, indices=all_indices, base=base, check_state=check_state, c_dtype=c_dtype)
    return vn_entropy_1 + vn_entropy_2 - vn_entropy_12