import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
def _reconstruct_xs(self, xs: Optional[np.ndarray]) -> np.ndarray:
    if xs is None:
        new_xs = np.zeros((2 * self.n + 1, self.n), dtype=bool)
        for i in range(self.n):
            new_xs[i, i] = True
    else:
        shape = xs.shape
        if len(shape) == 2 and shape[0] == 2 * self.n and (shape[1] == self.n) and (xs.dtype == np.dtype(bool)):
            new_xs = np.append(xs, np.zeros((1, self.n), dtype=bool), axis=0)
        else:
            raise ValueError("The value you passed for xs is not the correct shape and/or type. Please confirm that it's 2*num_qubits rows, num_qubits columns, and of type bool.")
    return new_xs