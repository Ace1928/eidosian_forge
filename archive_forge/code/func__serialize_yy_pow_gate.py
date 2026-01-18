import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_yy_pow_gate(self, gate: cirq.YYPowGate, targets: Sequence[int]) -> dict:
    return self._serialize_parity_pow_gate(gate, targets, 'yy')