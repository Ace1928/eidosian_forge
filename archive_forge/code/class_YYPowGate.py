from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Sequence
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
@value.value_equality
class YYPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """The Y-parity gate, possibly raised to a power.

    The YY**t gate implements the following unitary:

    $$
    (Y \\otimes Y)^t = \\begin{bmatrix}
                      c & 0 & 0 & -s \\\\
                      0 & c & s & 0 \\\\
                      0 & s & c & 0 \\\\
                      -s & 0 & 0 & c \\\\
                      \\end{bmatrix}
    $$

    where

    $$
    c = f \\cos\\left(\\frac{\\pi t}{2}\\right)
    $$

    $$
    s = -i f \\sin\\left(\\frac{\\pi t}{2}\\right)
    $$

    $$
    f = e^{\\frac{i \\pi t}{2}}.
    $$
    """

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0.0, np.array([[0.5, 0, 0, -0.5], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [-0.5, 0, 0, 0.5]])), (1.0, np.array([[0.5, 0, 0, 0.5], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0.5, 0, 0, 0.5]]))]

    def _eigen_shifts(self):
        return [0, 1]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _decompose_into_clifford_with_qubits_(self, qubits):
        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return [pauli_interaction_gate.PauliInteractionGate(pauli_gates.Y, False, pauli_gates.Y, False).on(*qubits), clifford_gate.SingleQubitCliffordGate.Y_sqrt.on_each(*qubits)]
        if self.exponent % 2 == 1:
            return [clifford_gate.SingleQubitCliffordGate.Y.on_each(*qubits)]
        if self.exponent % 2 == 1.5:
            return [pauli_interaction_gate.PauliInteractionGate(pauli_gates.Y, False, pauli_gates.Y, False).on(*qubits), clifford_gate.SingleQubitCliffordGate.Y_nsqrt.on_each(*qubits)]
        return NotImplemented

    def _has_stabilizer_effect_(self) -> bool:
        return self.exponent % 2 in (0, 0.5, 1, 1.5)

    def _decompose_(self, qubits: Tuple['cirq.Qid', ...]) -> 'cirq.OP_TREE':
        yield common_gates.XPowGate(exponent=0.5).on_each(*qubits)
        yield ZZPowGate(exponent=self.exponent, global_shift=self.global_shift)(*qubits)
        yield common_gates.XPowGate(exponent=-0.5).on_each(*qubits)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=('YY', 'YY'), exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'YY'
        return f'YY**{self._exponent!r}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.YY'
            return f'(cirq.YY**{proper_repr(self._exponent)})'
        return f'cirq.YYPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'