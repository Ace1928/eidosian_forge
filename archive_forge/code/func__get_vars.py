import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
def _get_vars(self, qubit_pattern: Sequence['cirq.Qid']) -> List[int]:
    in_vars = [2 * self._qubits_to_idx[q] for q in qubit_pattern]
    out_vars = [2 * self._qubits_to_idx[q] + 1 for q in qubit_pattern]
    return in_vars + out_vars