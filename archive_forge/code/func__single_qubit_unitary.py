import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def _single_qubit_unitary(theta: _RealArraylike, phi_d: _RealArraylike, phi_o: _RealArraylike) -> np.ndarray:
    """Single qubit unitary matrix.

    Args:
        theta: cos(theta) is magnitude of 00 matrix element. May be a scalar
           or real ndarray (for broadcasting).
        phi_d: exp(i phi_d) is the phase of 00 matrix element. May be a scalar
           or real ndarray (for broadcasting).
        phi_o: i exp(i phi_o) is the phase of 10 matrix element. May be a scalar
           or real ndarray (for broadcasting).


    Notes:
        The output is vectorized with respect to the angles. I.e, if the angles
        are (broadcastable) arraylike objects whose sum would have shape (...),
        the output is an array of shape (...,2,2), where the final two indices
        correspond to unitary matrices.
    """
    U00 = np.cos(theta) * np.exp(1j * np.asarray(phi_d))
    U10 = 1j * np.sin(theta) * np.exp(1j * np.asarray(phi_o))
    Udiag = np.array([[U00, np.zeros_like(U00)], [np.zeros_like(U00), U00.conj()]])
    Udiag = np.moveaxis(Udiag, [0, 1], [-2, -1])
    Uoff = np.array([[np.zeros_like(U10), -U10.conj()], [U10, np.zeros_like(U10)]])
    Uoff = np.moveaxis(Uoff, [0, 1], [-2, -1])
    return Udiag + Uoff