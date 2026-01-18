from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
@value.value_equality()
class ThreeQubitDiagonalGate(raw_types.Gate):
    """A three qubit gate whose unitary is given by a diagonal $8 \\times 8$ matrix.

    This gate's off-diagonal elements are zero and its on diagonal
    elements are all phases.
    """

    def __init__(self, diag_angles_radians: Sequence[value.TParamVal]) -> None:
        """A three qubit gate with only diagonal elements.

        This gate's off-diagonal elements are zero and its on diagonal
        elements are all phases.

        Args:
            diag_angles_radians: The list of angles on the diagonal in radians.
                If these values are $(x_0, x_1, \\ldots , x_7)$ then the unitary
                has diagonal values $(e^{i x_0}, e^{i x_1}, \\ldots, e^{i x_7})$.
        """
        self._diag_angles_radians: Tuple[value.TParamVal, ...] = tuple(diag_angles_radians)

    @property
    def diag_angles_radians(self) -> Tuple[value.TParamVal, ...]:
        return self._diag_angles_radians

    def _is_parameterized_(self) -> bool:
        return any((protocols.is_parameterized(angle) for angle in self._diag_angles_radians))

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for angle in self._diag_angles_radians for name in protocols.parameter_names(angle)}

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'ThreeQubitDiagonalGate':
        return self.__class__([protocols.resolve_parameters(angle, resolver, recursive) for angle in self._diag_angles_radians])

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        return np.diag([np.exp(1j * angle) for angle in self._diag_angles_radians])

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        for index, angle in enumerate(self._diag_angles_radians):
            little_endian_index = 4 * (index & 1) + 2 * (index >> 1 & 1) + (index >> 2 & 1)
            subspace_index = args.subspace_index(little_endian_index)
            args.target_tensor[subspace_index] *= np.exp(1j * angle)
        return args.target_tensor

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        rounded_angles = np.array(self._diag_angles_radians)
        if args.precision is not None:
            rounded_angles = rounded_angles.round(args.precision)
        diag_str = f'diag({', '.join((proper_repr(angle) for angle in rounded_angles))})'
        return protocols.CircuitDiagramInfo((diag_str, '#2', '#3'))

    def __pow__(self, exponent: Any) -> 'ThreeQubitDiagonalGate':
        if not isinstance(exponent, (int, float, sympy.Basic)):
            return NotImplemented
        return ThreeQubitDiagonalGate([protocols.mul(angle, exponent, NotImplemented) for angle in self._diag_angles_radians])

    def _decompose_(self, qubits):
        """An adjacency-respecting decomposition.

        0: ───p_0───@──────────────@───────@──────────@──────────
                    │              │       │          │
        1: ───p_1───X───@───p_3────X───@───X──────@───X──────@───
                        │              │          │          │
        2: ───p_2───────X───p_4────────X───p_5────X───p_6────X───

        where p_i = T**(4*x_i) and x_i solve the system of equations
                    [0, 0, 1, 0, 1, 1, 1][x_0]   [r_1]
                    [0, 1, 0, 1, 1, 0, 1][x_1]   [r_2]
                    [0, 1, 1, 1, 0, 1, 0][x_2]   [r_3]
                    [1, 0, 0, 1, 1, 1, 0][x_3] = [r_4]
                    [1, 0, 1, 1, 0, 0, 1][x_4]   [r_5]
                    [1, 1, 0, 0, 0, 1, 1][x_5]   [r_6]
                    [1, 1, 1, 0, 1, 0, 0][x_6]   [r_7]
        where r_i is self._diag_angles_radians[i].

        The above system was created by equating the composition of the gates
        in the circuit diagram to np.diag(self._diag_angles) (shifted by a
        global phase of np.exp(-1j * self._diag_angles[0])).
        """
        a, b, c = qubits
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = (c, b)
            elif not b.is_adjacent(c):
                a, b = (b, a)
        sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]
        phase_matrix_inverse = 0.25 * np.array([[-1, -1, -1, 1, 1, 1, 1], [-1, 1, 1, -1, -1, 1, 1], [1, -1, 1, -1, 1, -1, 1], [-1, 1, 1, 1, 1, -1, -1], [1, 1, -1, 1, -1, -1, 1], [1, -1, 1, 1, -1, 1, -1], [1, 1, -1, -1, 1, 1, -1]])
        shifted_angles_tail = [angle - self._diag_angles_radians[0] for angle in self._diag_angles_radians[1:]]
        phase_solutions = phase_matrix_inverse.dot(shifted_angles_tail)
        p_gates = [pauli_gates.Z ** (solution / np.pi) for solution in phase_solutions]
        global_phase = 1j ** (2 * self._diag_angles_radians[0] / np.pi)
        global_phase_operation = [global_phase_op.global_phase_operation(global_phase)] if protocols.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0 else []
        return global_phase_operation + [p_gates[0](a), p_gates[1](b), p_gates[2](c), sweep_abc, p_gates[3](b), p_gates[4](c), sweep_abc, p_gates[5](c), sweep_abc, p_gates[6](c), sweep_abc]

    def _value_equality_values_(self):
        return tuple(self._diag_angles_radians)

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        x = [np.exp(1j * angle) for angle in self._diag_angles_radians]
        return value.LinearDict({'III': (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]) / 8, 'IIZ': (x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7]) / 8, 'IZI': (x[0] + x[1] - x[2] - x[3] + x[4] + x[5] - x[6] - x[7]) / 8, 'IZZ': (x[0] - x[1] - x[2] + x[3] + x[4] - x[5] - x[6] + x[7]) / 8, 'ZII': (x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7]) / 8, 'ZIZ': (x[0] - x[1] + x[2] - x[3] - x[4] + x[5] - x[6] + x[7]) / 8, 'ZZI': (x[0] + x[1] - x[2] - x[3] - x[4] - x[5] + x[6] + x[7]) / 8, 'ZZZ': (x[0] - x[1] - x[2] + x[3] - x[4] + x[5] + x[6] - x[7]) / 8})

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=['diag_angles_radians'])

    def __repr__(self) -> str:
        angles = ','.join((proper_repr(angle) for angle in self._diag_angles_radians))
        return f'cirq.ThreeQubitDiagonalGate([{angles}])'

    def _num_qubits_(self) -> int:
        return 3