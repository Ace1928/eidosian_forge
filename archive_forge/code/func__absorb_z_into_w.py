from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _absorb_z_into_w(op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal]) -> 'cirq.OP_TREE':
    """Absorbs a Z^t gate into a W(a) flip.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───Z^t───
        ≡ ───W(a)───────────Z^t/2──────────Z^t/2─── (split Z)
        ≡ ───W(a)───W(a)───Z^-t/2───W(a)───Z^t/2─── (flip Z)
        ≡ ───W(a)───W(a)──────────W(a+t/2)───────── (phase W)
        ≡ ────────────────────────W(a+t/2)───────── (cancel Ws)
        ≡ ───W(a+t/2)───
    """
    t = cast(value.TParamVal, _try_get_known_z_half_turns(op))
    q = op.qubits[0]
    held_w_phases[q] += t / 2
    return []