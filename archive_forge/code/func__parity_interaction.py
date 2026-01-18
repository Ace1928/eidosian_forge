from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def _parity_interaction(q0: 'cirq.Qid', q1: 'cirq.Qid', rads: float, atol: float, gate: Optional[ops.Gate]=None):
    """Yields a ZZ interaction framed by the given operation."""
    if abs(rads) < atol:
        return
    h = rads * -2 / np.pi
    if gate is not None:
        yield (gate.on(q0), gate.on(q1))
    if _is_trivial_angle(rads, atol):
        yield ops.CZ.on(q0, q1)
    else:
        yield (ops.CZ(q0, q1) ** (-2 * h))
    yield (ops.Z(q0) ** h)
    yield (ops.Z(q1) ** h)
    if gate is not None:
        g = protocols.inverse(gate)
        yield (g.on(q0), g.on(q1))