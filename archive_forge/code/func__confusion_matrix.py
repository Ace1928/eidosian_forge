import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
def _confusion_matrix(self, qubits: Sequence['cirq.Qid']) -> np.ndarray:
    ein_input = []
    for qs, cm in zip(self.measure_qubits, self.confusion_matrices):
        ein_input.extend([cm.reshape((2, 2) * len(qs)), self._get_vars(qs)])
    ein_out = self._get_vars(qubits)
    ret = np.einsum(*ein_input, ein_out).reshape((2 ** len(qubits),) * 2)
    return ret / ret.sum(axis=1)