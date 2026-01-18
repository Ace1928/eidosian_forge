import warnings
import numpy as np
from .layer import (
from .pmatrix import PMatrix
from ..cnot_unit_objective import CNOTUnitObjective
def _calc_objective_function(self) -> float:
    """
        Computes the value of objective function.
        """
    ucf = self._ucf_mat.finalize(temp_mat=self._tmp1)
    trace_ucf = np.trace(ucf)
    fobj = abs(2 ** self._num_qubits - float(np.real(trace_ucf)))
    return fobj