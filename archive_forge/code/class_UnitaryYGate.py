from typing import Tuple
import numpy as np
import pytest
import cirq
class UnitaryYGate(cirq.Gate):

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,)

    def _unitary_(self):
        return np.array([[0, -1j], [1j, 0]])