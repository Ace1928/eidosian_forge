from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def _non_local_part(q0: 'cirq.Qid', q1: 'cirq.Qid', interaction_coefficients: Tuple[float, float, float], allow_partial_czs: bool, atol: float=1e-08):
    """Yields non-local operation of KAK decomposition."""
    x, y, z = interaction_coefficients
    if allow_partial_czs or all((_is_trivial_angle(e, atol) for e in [x, y, z])):
        return [_parity_interaction(q0, q1, x, atol, ops.Y ** (-0.5)), _parity_interaction(q0, q1, y, atol, ops.X ** 0.5), _parity_interaction(q0, q1, z, atol)]
    if abs(z) >= atol:
        return _xx_yy_zz_interaction_via_full_czs(q0, q1, x, y, z)
    if y >= atol:
        return _xx_yy_interaction_via_full_czs(q0, q1, x, y)
    return _xx_interaction_via_full_czs(q0, q1, x)