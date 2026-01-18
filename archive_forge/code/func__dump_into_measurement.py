from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _dump_into_measurement(op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal]) -> 'cirq.OP_TREE':
    measurement = cast(ops.MeasurementGate, cast(ops.GateOperation, op).gate)
    new_measurement = measurement.with_bits_flipped(*[i for i, q in enumerate(op.qubits) if q in held_w_phases]).on(*op.qubits)
    for q in op.qubits:
        held_w_phases.pop(q, None)
    return new_measurement