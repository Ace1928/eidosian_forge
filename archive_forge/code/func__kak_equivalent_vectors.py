import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def _kak_equivalent_vectors(kak_vec) -> np.ndarray:
    """Generates all KAK vectors equivalent under single qubit unitaries."""
    kak_vec = np.asarray(kak_vec, dtype=float)
    out = np.einsum('pab,...b->...pa', _perms_123, kak_vec)
    out = np.einsum('nab,...b->...na', _negations, out)
    out = out[..., np.newaxis, :, :, :] + _offsets[:, np.newaxis, np.newaxis, :]
    return np.reshape(out, out.shape[:-4] + (192, 3))