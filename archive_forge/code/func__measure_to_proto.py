import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _measure_to_proto(gate: cirq.MeasurementGate, qubits: Sequence[cirq.Qid]):
    if len(qubits) == 0:
        raise ValueError('Measurement gate on no qubits.')
    invert_mask = None
    if gate.invert_mask:
        invert_mask = gate.invert_mask + (False,) * (gate.num_qubits() - len(gate.invert_mask))
    if invert_mask and len(invert_mask) != len(qubits):
        raise ValueError('Measurement gate had invert mask of length different than number of qubits it acts on.')
    return operations_pb2.Measurement(targets=[_qubit_to_proto(q) for q in qubits], key=cirq.measurement_key_name(gate), invert_mask=invert_mask)