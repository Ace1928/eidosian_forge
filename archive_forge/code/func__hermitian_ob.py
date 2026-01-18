from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _hermitian_ob(self, observable, wires_map: dict):
    """Serializes a Hermitian observable"""
    assert not isinstance(observable, Tensor)
    wires = [wires_map[w] for w in observable.wires]
    return self.hermitian_obs(matrix(observable).ravel().astype(self.ctype), wires)