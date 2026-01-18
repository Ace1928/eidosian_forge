import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
class _ZEigenState(_PauliEigenState):
    _symbol = 'Z'

    def state_vector(self) -> np.ndarray:
        if self.eigenvalue == 1:
            return np.array([1, 0])
        elif self.eigenvalue == -1:
            return np.array([0, 1])
        raise ValueError(f'Bad eigenvalue: {self.eigenvalue}')

    def stabilized_by(self) -> Tuple[int, 'cirq.Pauli']:
        from cirq import ops
        return (self.eigenvalue, ops.Z)