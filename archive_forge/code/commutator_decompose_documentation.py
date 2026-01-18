from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
Decompose an :math:`SO(3)`-matrix, :math:`U` as a balanced commutator.

    This function finds two :math:`SO(3)` matrices :math:`V, W` such that the input matrix
    equals

    .. math::

        U = V^\dagger W^\dagger V W.

    For this decomposition, the following statement holds


    .. math::

        ||V - I||_F, ||W - I||_F \leq \frac{\sqrt{||U - I||_F}}{2},

    where :math:`I` is the identity and :math:`||\cdot ||_F` is the Frobenius norm.

    Args:
        u_so3: SO(3)-matrix that needs to be decomposed as balanced commutator.
        check_input: If True, checks whether the input matrix is actually SO(3).

    Returns:
        Tuple of GateSequences from SO(3)-matrices :math:`V, W`.

    Raises:
        ValueError: if ``u_so3`` is not an SO(3)-matrix.
    