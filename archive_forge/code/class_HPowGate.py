from typing import (
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import controlled_gate, eigen_gate, gate_features, raw_types, control_values as cv
from cirq.type_workarounds import NotImplementedType
from cirq.ops.swap_gates import ISWAP, SWAP, ISwapPowGate, SwapPowGate
from cirq.ops.measurement_gate import MeasurementGate
imports.
class HPowGate(eigen_gate.EigenGate):
    """A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

    The unitary matrix of `cirq.HPowGate(exponent=t)` is:
    $$
        \\begin{bmatrix}
            e^{i\\pi t/2} \\left(\\cos(\\pi t/2) - i \\frac{\\sin (\\pi t /2)}{\\sqrt{2}}\\right)
                && -i e^{i\\pi t/2} \\frac{\\sin(\\pi t /2)}{\\sqrt{2}} \\\\
            -i e^{i\\pi t/2} \\frac{\\sin(\\pi t /2)}{\\sqrt{2}}
                && e^{i\\pi t/2} \\left(\\cos(\\pi t/2) + i \\frac{\\sin (\\pi t /2)}{\\sqrt{2}}\\right)
        \\end{bmatrix}
    $$
    Note in particular that for $t=1$, this gives the Hadamard matrix
    $$
        \\begin{bmatrix}
            \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\\
            \\frac{1}{\\sqrt{2}} & -\\frac{1}{\\sqrt{2}}
        \\end{bmatrix}
    $$

    `cirq.H`, the Hadamard gate, is an instance of this gate at `exponent=1`.
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        s = np.sqrt(2)
        component0 = np.array([[3 + 2 * s, 1 + s], [1 + s, 1]]) / (4 + 2 * s)
        component1 = np.array([[3 - 2 * s, 1 - s], [1 - s, 1]]) / (4 - 2 * s)
        return [(0, component0), (1, component1)]

    def _num_qubits_(self) -> int:
        return 1

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        angle = np.pi * self._exponent / 2
        return value.LinearDict({'I': phase * np.cos(angle), 'X': -1j * phase * np.sin(angle) / np.sqrt(2), 'Z': -1j * phase * np.sin(angle) / np.sqrt(2)})

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate
        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.H.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.target_tensor[one] -= args.target_tensor[zero]
        args.target_tensor[one] *= -0.5
        args.target_tensor[zero] -= args.target_tensor[one]
        p = 1j ** (2 * self._exponent * self._global_shift)
        args.target_tensor *= np.sqrt(2) * p
        return args.target_tensor

    def _decompose_(self, qubits):
        q = qubits[0]
        if self._exponent == 1:
            yield (cirq.Y(q) ** 0.5)
            yield cirq.XPowGate(global_shift=-0.25 + self.global_shift).on(q)
            return
        yield YPowGate(exponent=0.25).on(q)
        yield XPowGate(exponent=self._exponent, global_shift=self.global_shift).on(q)
        yield YPowGate(exponent=-0.25).on(q)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=('H',), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 0:
            return args.format('id {0};\n', qubits[0])
        elif self._exponent == 1 and self._global_shift == 0:
            return args.format('h {0};\n', qubits[0])
        return args.format('ry({0:half_turns}) {3};\nrx({1:half_turns}) {3};\nry({2:half_turns}) {3};\n', 0.25, self._exponent, -0.25, qubits[0])

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'H'
        return f'H**{self._exponent}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.H'
            return f'(cirq.H**{proper_repr(self._exponent)})'
        return f'cirq.HPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'