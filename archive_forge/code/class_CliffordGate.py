from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@value.value_equality
class CliffordGate(raw_types.Gate, CommonCliffordGates):
    """Clifford rotation for N-qubit."""

    def __init__(self, *, _clifford_tableau: qis.CliffordTableau) -> None:
        self._clifford_tableau = _clifford_tableau.copy()

    @property
    def clifford_tableau(self):
        return self._clifford_tableau

    def _json_dict_(self) -> Dict[str, Any]:
        json_dict = self._clifford_tableau._json_dict_()
        return json_dict

    def _value_equality_values_(self):
        return self._clifford_tableau.matrix().tobytes() + self._clifford_tableau.rs.tobytes()

    def _num_qubits_(self):
        return self.clifford_tableau.n

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return True

    def __pow__(self, exponent) -> 'CliffordGate':
        if exponent == -1:
            return CliffordGate.from_clifford_tableau(self.clifford_tableau.inverse())
        if exponent > 0 and int(exponent) == exponent:
            base_tableau = self.clifford_tableau.copy()
            for _ in range(int(exponent) - 1):
                base_tableau = base_tableau.then(self.clifford_tableau)
            return CliffordGate.from_clifford_tableau(base_tableau)
        if exponent < 0 and int(exponent) == exponent:
            base_tableau = self.clifford_tableau.copy()
            for _ in range(int(-exponent) - 1):
                base_tableau = base_tableau.then(self.clifford_tableau)
            return CliffordGate.from_clifford_tableau(base_tableau.inverse())
        return NotImplemented

    def __repr__(self) -> str:
        return f'Clifford Gate with Tableau:\n {self.clifford_tableau._str_full_()}'

    def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType, None]:
        return NotImplemented

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return transformers.analytical_decompositions.decompose_clifford_tableau_to_operations(list(qubits), self.clifford_tableau)

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']) -> Union[NotImplementedType, bool]:
        if isinstance(sim_state, sim.clifford.CliffordTableauSimulationState):
            axes = sim_state.get_axes(qubits)
            padded_tableau = _pad_tableau(self._clifford_tableau, len(sim_state.qubits), axes)
            sim_state._state = sim_state.tableau.then(padded_tableau)
            return True
        if isinstance(sim_state, sim.clifford.StabilizerChFormSimulationState):
            return NotImplemented
        return NotImplemented