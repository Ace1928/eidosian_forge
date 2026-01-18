from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _potential_cross_whole_w(op: ops.Operation, atol: float, held_w_phases: Dict[ops.Qid, value.TParamVal]) -> 'cirq.OP_TREE':
    """Grabs or cancels a held W gate against an existing W gate.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───W(b)───
        ≡ ───Z^-a───X───Z^a───Z^-b───X───Z^b───
        ≡ ───Z^-a───Z^-a───Z^b───X───X───Z^b───
        ≡ ───Z^-a───Z^-a───Z^b───Z^b───
        ≡ ───Z^2(b-a)───
    """
    _, phase_exponent = cast(Tuple[value.TParamVal, value.TParamVal], _try_get_known_phased_pauli(op))
    q = op.qubits[0]
    a = held_w_phases.get(q, None)
    b = phase_exponent
    if a is None:
        held_w_phases[q] = b
    else:
        del held_w_phases[q]
        t = 2 * (b - a)
        if not single_qubit_decompositions.is_negligible_turn(t / 2, atol):
            return ops.Z(q) ** t
    return []