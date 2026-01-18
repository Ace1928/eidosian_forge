from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def _phase(exponent, global_shift):
    return np.exp(1j * np.pi * global_shift * exponent)