from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def _xx_interaction_via_full_czs(q0: 'cirq.Qid', q1: 'cirq.Qid', x: float):
    a = x * -2 / np.pi
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield (ops.X(q0) ** a)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)