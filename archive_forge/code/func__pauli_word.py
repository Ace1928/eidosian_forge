from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _pauli_word(self, observable, wires_map: dict):
    """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
    if len(observable) == 1:
        wire, pauli = list(observable.items())[0]
        return self.named_obs(pauli_name_map[pauli], [wires_map[wire]])
    return self.tensor_obs([self.named_obs(pauli_name_map[pauli], [wires_map[wire]]) for wire, pauli in observable.items()])