import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
def _reconstruct_rs(self, rs: Optional[np.ndarray]) -> np.ndarray:
    if rs is None:
        new_rs = np.zeros(2 * self.n + 1, dtype=bool)
        for i, val in enumerate(big_endian_int_to_digits(self.initial_state, digit_count=self.n, base=2)):
            new_rs[self.n + i] = bool(val)
    else:
        shape = rs.shape
        if len(shape) == 1 and shape[0] == 2 * self.n and (rs.dtype == np.dtype(bool)):
            new_rs = np.append(rs, np.zeros(1, dtype=bool))
        else:
            raise ValueError("The value you passed for rs is not the correct shape and/or type. Please confirm that it's a single row with 2*num_qubits columns and of type bool.")
    return new_rs