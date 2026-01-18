import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _validate_qubits(self, all_qubits: Collection['cirq.Qid']) -> int:
    """Returns the number of qubits while validating qubit types and values."""
    if any((not isinstance(q, line_qubit.LineQubit) for q in all_qubits)):
        raise ValueError(f'All qubits must be cirq.LineQubits but were {set((type(q) for q in all_qubits))}')
    if any((cast(line_qubit.LineQubit, q).x < 0 for q in all_qubits)):
        raise ValueError(f'IonQ API must use LineQubits from 0 to number of qubits - 1. Instead found line qubits with indices {all_qubits}.')
    num_qubits = cast(line_qubit.LineQubit, max(all_qubits)).x + 1
    return num_qubits