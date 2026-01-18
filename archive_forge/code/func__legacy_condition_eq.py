import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
def _legacy_condition_eq(cond1, cond2, bit_indices1, bit_indices2):
    if cond1 is cond2 is None:
        return True
    elif None in (cond1, cond2):
        return False
    target1, val1 = cond1
    target2, val2 = cond2
    if val1 != val2:
        return False
    if isinstance(target1, Clbit) and isinstance(target2, Clbit):
        return bit_indices1[target1] == bit_indices2[target2]
    if isinstance(target1, ClassicalRegister) and isinstance(target2, ClassicalRegister):
        return target1.size == target2.size and all((bit_indices1[t1] == bit_indices2[t2] for t1, t2 in zip(target1, target2)))
    return False