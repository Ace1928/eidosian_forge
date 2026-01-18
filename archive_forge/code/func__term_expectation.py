from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def _term_expectation(wf: np.ndarray, term: PauliTerm) -> Any:
    wf2 = wf
    for qubit_i, op_str in term._ops.items():
        assert isinstance(qubit_i, int)
        op_mat = QUANTUM_GATES[op_str]
        wf2 = targeted_tensordot(gate=op_mat, wf=wf2, wf_target_inds=[qubit_i])
    return cast(complex, term.coefficient) * np.tensordot(wf.conj(), wf2, axes=len(wf.shape))