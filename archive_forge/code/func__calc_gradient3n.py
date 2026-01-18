import warnings
import numpy as np
from .layer import (
from .pmatrix import PMatrix
from ..cnot_unit_objective import CNOTUnitObjective
def _calc_gradient3n(self, grad3n: np.ndarray):
    """
        Calculates a part gradient contributed by 1-qubit gates.
        """
    ucf = self._ucf_mat
    tmp1, tmp2 = (self._tmp1, self._tmp2)
    f_gates = self._f_gates
    f_dervs = self._f_dervs
    f_layers = self._f_layers
    for q in range(self._num_qubits):
        if q > 0:
            f_layers[q - 1].set_from_matrix(mat=f_gates[q - 1])
            ucf.mul_left_q1(f_layers[q - 1], temp_mat=tmp1)
        ucf.mul_right_q1(f_layers[q], temp_mat=tmp1, dagger=True)
        ucf.finalize(temp_mat=tmp1)
        for i in range(3):
            f_layers[q].set_from_matrix(mat=f_dervs[q, i])
            grad3n[q, i] = -1 * np.real(ucf.product_q1(layer=f_layers[q], tmp1=tmp1, tmp2=tmp2))