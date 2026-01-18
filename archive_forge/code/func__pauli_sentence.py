from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _pauli_sentence(self, observable, wires_map: dict):
    """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
    pwords, coeffs = zip(*observable.items())
    terms = [self._pauli_word(pw, wires_map) for pw in pwords]
    coeffs = np.array(coeffs).astype(self.rtype)
    if self.split_obs:
        return [self.hamiltonian_obs([c], [t]) for c, t in zip(coeffs, terms)]
    return self.hamiltonian_obs(coeffs, terms)