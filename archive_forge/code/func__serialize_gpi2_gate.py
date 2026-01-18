import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_gpi2_gate(self, gate: GPI2Gate, targets: Sequence[int]) -> Optional[dict]:
    return {'gate': 'gpi2', 'target': targets[0], 'phase': gate.phase}