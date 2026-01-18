from typing import Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
from cirq.ops import DensePauliString
from cirq import protocols
def _argmax(V: npt.NDArray) -> Tuple[int, float]:
    """Returns a tuple (index of max number, max number)."""
    V = (V * V.conj()).real
    idx_max = np.argmax(V)
    V[idx_max] = 0
    return (cast(int, idx_max), np.max(V))