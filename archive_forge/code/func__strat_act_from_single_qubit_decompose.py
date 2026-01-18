import abc
from typing import Any, cast, Generic, Optional, Sequence, TYPE_CHECKING, TypeVar, Union
import numpy as np
import sympy
from cirq import linalg, ops, protocols
from cirq.ops import common_gates, global_phase_op, matrix_gates, swap_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.simulation_state import SimulationState
from cirq.type_workarounds import NotImplementedType
def _strat_act_from_single_qubit_decompose(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
    if num_qubits(val) == 1:
        if not has_unitary(val):
            return NotImplemented
        u = unitary(val)
        gate_and_phase = SingleQubitCliffordGate.from_unitary_with_global_phase(u)
        if gate_and_phase is not None:
            clifford_gate, global_phase = gate_and_phase
            for gate in clifford_gate.decompose_gate():
                self._strat_apply_gate(gate, qubits)
            self._state.apply_global_phase(global_phase)
            return True
    return NotImplemented