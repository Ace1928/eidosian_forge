from typing import Any, Iterator, Tuple, Union, TYPE_CHECKING
import numpy as np
import sympy
from cirq import linalg, protocols, value, _compat
from cirq.ops import linear_combinations, pauli_string_phasor
def _all_pauli_strings_commute(pauli_sum: 'cirq.PauliSum') -> bool:
    for x in pauli_sum:
        for y in pauli_sum:
            if not protocols.commutes(x, y):
                return False
    return True