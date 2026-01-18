from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def _CNOT_right(self, q, r):
    """Right multiplication version of CNOT gate."""
    self.G[:, q] ^= self.G[:, r]
    self.F[:, r] ^= self.F[:, q]
    self.M[:, q] ^= self.M[:, r]