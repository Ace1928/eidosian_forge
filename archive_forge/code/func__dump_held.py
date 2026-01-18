from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _dump_held(qubits: Iterable[ops.Qid], held_w_phases: Dict[ops.Qid, value.TParamVal]) -> 'cirq.OP_TREE':
    for q in sorted(qubits):
        p = held_w_phases.get(q)
        if p is not None:
            dump_op = ops.PhasedXPowGate(phase_exponent=p).on(q)
            yield dump_op
        held_w_phases.pop(q, None)