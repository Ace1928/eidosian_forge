import warnings
import numpy as np
from .layer import (
from .pmatrix import PMatrix
from ..cnot_unit_objective import CNOTUnitObjective
def _init_layers(self):
    """
        Initializes C-layers and F-layers by corresponding gate matrices.
        """
    c_gates = self._c_gates
    c_layers = self._c_layers
    for q in range(self.num_cnots):
        c_layers[q].set_from_matrix(mat=c_gates[q])
    f_gates = self._f_gates
    f_layers = self._f_layers
    for q in range(self._num_qubits):
        f_layers[q].set_from_matrix(mat=f_gates[q])