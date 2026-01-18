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
def _check_state_vector(state_vector):
    """Check the shape and the norm of a state vector."""
    dim = state_vector.shape[-1]
    if len(np.shape(state_vector)) not in (1, 2) or not np.log2(dim).is_integer():
        raise ValueError('State vector must be of shape (2**wires,) or (batch_dim, 2**wires)')
    if len(state_vector.shape) == 1:
        state_vector = qml.math.stack([state_vector])
    if not is_abstract(state_vector):
        for sv in state_vector:
            norm = np.linalg.norm(sv, ord=2)
            if not allclose(norm, 1.0, atol=1e-10):
                raise ValueError('Sum of amplitudes-squared does not equal one.')