from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _hamiltonian(self, observable, wires_map: dict):
    coeffs = np.array(unwrap(observable.coeffs)).astype(self.rtype)
    terms = [self._ob(t, wires_map) for t in observable.ops]
    if self.split_obs:
        return [self.hamiltonian_obs([c], [t]) for c, t in zip(coeffs, terms)]
    return self.hamiltonian_obs(coeffs, terms)