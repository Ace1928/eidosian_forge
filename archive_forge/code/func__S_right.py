from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def _S_right(self, q):
    """Right multiplication version of S gate."""
    self.M[:, q] ^= self.F[:, q]
    self.gamma[:] = (self.gamma[:] - self.F[:, q]) % 4