from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _ob(self, observable, wires_map):
    """Serialize a :class:`pennylane.operation.Observable` into an Observable."""
    if isinstance(observable, Tensor):
        return self._tensor_ob(observable, wires_map)
    if observable.name == 'Hamiltonian':
        return self._hamiltonian(observable, wires_map)
    if observable.name == 'SparseHamiltonian':
        return self._sparse_hamiltonian(observable, wires_map)
    if isinstance(observable, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
        return self._named_obs(observable, wires_map)
    if observable._pauli_rep is not None:
        return self._pauli_sentence(observable._pauli_rep, wires_map)
    return self._hermitian_ob(observable, wires_map)