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
@value.value_equality
class YPowGate(eigen_gate.EigenGate):
    """A gate that rotates around the Y axis of the Bloch sphere.

    The unitary matrix of `cirq.YPowGate(exponent=t)` is:
    $$
        \\begin{bmatrix}
            e^{i \\pi t /2} \\cos(\\pi t /2) & - e^{i \\pi t /2} \\sin(\\pi t /2) \\\\
            e^{i \\pi t /2} \\sin(\\pi t /2) & e^{i \\pi t /2} \\cos(\\pi t /2)
        \\end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i \\pi t / 2}$ vs the traditionally defined rotation matrices
    about the Pauli Y axis. See `cirq.Ry` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.Y`, the Pauli Y gate, is an instance of this gate at `exponent=1`.

    Unlike `cirq.XPowGate` and `cirq.ZPowGate`, this gate has no generalization
    to qudits and hence does not take the dimension argument. Ignoring the
    global phase all generalized Pauli operators on a d-level system may be
    written as X**a Z**b for a,b=0,1,...,d-1. For a qubit, there is only one
    "mixed" operator: XZ, conventionally denoted -iY. However, when d > 2 there
    are (d-1)*(d-1) > 1 such "mixed" operators (still ignoring the global phase).
    Due to this ambiguity, qudit Y gate is not well defined. The "mixed" operators
    for qudits are generally not referred to by name, but instead are specified in
    terms of X and Z.
    """

    def _num_qubits_(self) -> int:
        return 1

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.available_buffer[zero] = -1j * args.target_tensor[one]
        args.available_buffer[one] = 1j * args.target_tensor[zero]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.available_buffer *= p
        return args.available_buffer

    def in_su2(self) -> 'Ry':
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Ry(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> 'YPowGate':
        """Returns an equal-up-global-phase standardized form of the gate."""
        return YPowGate(exponent=self._exponent)

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate
        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return SingleQubitCliffordGate.Y_sqrt.on(*qubits)
        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.Y.on(*qubits)
        if self.exponent % 2 == 1.5:
            return SingleQubitCliffordGate.Y_nsqrt.on(*qubits)
        return NotImplemented

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.array([[0.5, -0.5j], [0.5j, 0.5]])), (1, np.array([[0.5, 0.5j], [-0.5j, 0.5]]))]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        angle = np.pi * self._exponent / 2
        return value.LinearDict({'I': phase * np.cos(angle), 'Y': -1j * phase * np.sin(angle)})

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(wire_symbols=('Y',), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1 and self.global_shift != -0.5:
            return args.format('y {0};\n', qubits[0])
        return args.format('ry({0:half_turns}) {1};\n', self._exponent, qubits[0])

    @property
    def phase_exponent(self):
        return 0.5

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return cirq.ops.phased_x_gate.PhasedXPowGate(exponent=self._exponent, phase_exponent=0.5 + phase_turns * 2)

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 0.5 == 0

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'Y'
            return f'Y**{self._exponent}'
        return f'YPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.Y'
            return f'(cirq.Y**{proper_repr(self._exponent)})'
        return f'cirq.YPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'