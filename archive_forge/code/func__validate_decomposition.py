from typing import Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
from cirq.ops import DensePauliString
from cirq import protocols
def _validate_decomposition(decomposition: DensePauliString, U: npt.NDArray, eps: float) -> bool:
    """Returns whether the max absolute value of the elementwise difference is less than eps."""
    got = protocols.unitary(decomposition)
    return np.abs(got - U).max() < eps