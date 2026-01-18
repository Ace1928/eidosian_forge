import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_op(self, op: cirq.Operation) -> dict:
    if op.gate is None:
        raise ValueError(f'Attempt to serialize circuit with an operation which does not have a gate. Type: {type(op)} Op: {op}.')
    targets = [cast(line_qubit.LineQubit, q).x for q in op.qubits]
    gate = op.gate
    if cirq.is_parameterized(gate):
        raise ValueError(f'IonQ API does not support parameterized gates. Gate {gate} was parameterized. Consider resolving before sending.')
    gate_type = type(gate)
    for gate_mro_type in gate_type.mro():
        if gate_mro_type in self._dispatch:
            serialized_op = self._dispatch[gate_mro_type](gate, targets)
            if serialized_op:
                return serialized_op
    raise ValueError(f'Gate {gate} acting on {targets} cannot be serialized by IonQ API.')