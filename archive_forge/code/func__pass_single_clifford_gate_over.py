import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _pass_single_clifford_gate_over(pauli_map: Dict[TKey, pauli_gates.Pauli], gate: clifford_gate.SingleQubitCliffordGate, qubit: TKey, after_to_before: bool=False) -> bool:
    if qubit not in pauli_map:
        return False
    if not after_to_before:
        gate **= -1
    pauli, inv = gate.pauli_tuple(pauli_map[qubit])
    pauli_map[qubit] = pauli
    return inv