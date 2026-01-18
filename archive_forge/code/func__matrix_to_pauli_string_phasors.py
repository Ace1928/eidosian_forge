from typing import List, Union, Type, cast, TYPE_CHECKING
from enum import Enum
import numpy as np
from cirq import ops, transformers, protocols, linalg
from cirq.type_workarounds import NotImplementedType
def _matrix_to_pauli_string_phasors(mat: np.ndarray, qubit: 'cirq.Qid', *, keep_clifford: bool, atol: float) -> ops.OP_TREE:
    rotations = transformers.single_qubit_matrix_to_pauli_rotations(mat, atol)
    out_ops: List[ops.GateOperation] = []
    for pauli, half_turns in rotations:
        if keep_clifford and linalg.all_near_zero_mod(half_turns, 0.5):
            cliff_gate = ops.SingleQubitCliffordGate.from_quarter_turns(pauli, round(half_turns * 2))
            if out_ops and (not isinstance(out_ops[-1], ops.PauliStringPhasor)):
                gate = cast(ops.SingleQubitCliffordGate, out_ops[-1].gate)
                out_ops[-1] = gate.merged_with(cliff_gate)(qubit)
            else:
                out_ops.append(cliff_gate(qubit))
        else:
            out_ops.append(ops.PauliStringPhasor(ops.PauliString(pauli.on(qubit)), exponent_neg=round(half_turns, 10)))
    return out_ops