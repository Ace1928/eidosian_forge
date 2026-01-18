import numpy as np
import pennylane as qml
from pennylane import math
def _convert_to_su2(U, return_global_phase=False):
    """Convert a 2x2 unitary matrix to :math:`SU(2)`. (batched operation)

    Args:
        U (array[complex]): A matrix with a batch dimension, presumed to be
            of shape :math:`n \\times 2 \\times 2` and unitary for any positive integer n.
        return_global_phase (bool): If `True`, the return will include the global phase.
            If `False`, only the :math:`SU(2)` representation is returned.

    Returns:
        array[complex]: A :math:`n \\times 2 \\times 2` matrix in :math:`SU(2)` that is
            equivalent to U up to a global phase. If ``return_global_phase=True``, a
            2-element tuple is returned, with the first element being the :math:`SU(2)`
            equivalent and the second, the global phase.
    """
    U = qml.math.cast(U, 'complex128')
    determinants = math.linalg.det(U)
    phase = math.angle(determinants) / 2
    U = math.cast_like(U, determinants) * math.exp(-1j * math.cast_like(phase, 1j))[:, None, None]
    return (U, phase) if return_global_phase else U