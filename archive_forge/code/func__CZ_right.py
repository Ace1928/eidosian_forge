from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def _CZ_right(self, q, r):
    """Right multiplication version of CZ gate."""
    self.M[:, q] ^= self.F[:, r]
    self.M[:, r] ^= self.F[:, q]
    self.gamma[:] = (self.gamma[:] + 2 * self.F[:, q] * self.F[:, r]) % 4