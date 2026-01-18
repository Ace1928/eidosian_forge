from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _potential_cross_partial_w(op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal]) -> 'cirq.OP_TREE':
    """Cross the held W over a partial W gate.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───W(b)^t───
        ≡ ───Z^-a───X───Z^a───W(b)^t────── (expand W(a))
        ≡ ───Z^-a───X───W(b-a)^t───Z^a──── (move Z^a across, phasing axis)
        ≡ ───Z^-a───W(a-b)^t───X───Z^a──── (move X across, negating axis angle)
        ≡ ───W(2a-b)^t───Z^-a───X───Z^a─── (move Z^-a across, phasing axis)
        ≡ ───W(2a-b)^t───W(a)───
    """
    a = held_w_phases.get(op.qubits[0], None)
    if a is None:
        return op
    exponent, phase_exponent = cast(Tuple[value.TParamVal, value.TParamVal], _try_get_known_phased_pauli(op))
    new_op = ops.PhasedXPowGate(exponent=exponent, phase_exponent=2 * a - phase_exponent).on(op.qubits[0])
    return new_op