import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_cnot_pow_gate(self, gate: cirq.CNotPowGate, targets: Sequence[int]) -> Optional[dict]:
    if self._near_mod_n(gate.exponent, 1, 2):
        return {'gate': 'cnot', 'control': targets[0], 'target': targets[1]}
    return None