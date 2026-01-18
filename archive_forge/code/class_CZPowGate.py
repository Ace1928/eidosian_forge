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
class CZPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """A gate that applies a phase to the |11⟩ state of two qubits.

    The unitary matrix of `CZPowGate(exponent=t)` is:

    $$
    \\begin{bmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & e^{i \\pi t} \\\\
    \\end{bmatrix}
    $$

    `cirq.CZ`, the controlled Z gate, is an instance of this gate at
    `exponent=1`.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate
        if self.exponent % 2 == 1:
            return PauliInteractionGate.CZ.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 1, 1, 0])), (1, np.diag([0, 0, 0, 1]))]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Union[np.ndarray, NotImplementedType]:
        if protocols.is_parameterized(self):
            return NotImplemented
        c = 1j ** (2 * self._exponent)
        one_one = args.subspace_index(3)
        args.target_tensor[one_one] *= c
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        z_phase = 1j ** self._exponent
        c = -1j * z_phase * np.sin(np.pi * self._exponent / 2) / 2
        return value.LinearDict({'II': global_phase * (1 - c), 'IZ': global_phase * c, 'ZI': global_phase * c, 'ZZ': global_phase * -c})

    def _phase_by_(self, phase_turns, qubit_index):
        return self

    def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> raw_types.Gate:
        """Returns a controlled `CZPowGate`, using a `CCZPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CCZPowGate` or a `ControlledGate` of a `CCZPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `CZPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CCZPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CCZPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CCZPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CCZPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CCZPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        """
        if control_values and (not isinstance(control_values, cv.AbstractControlValues)):
            control_values = cv.ProductOfSums(tuple(((val,) if isinstance(val, int) else tuple(sorted(val)) for val in control_values)))
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if self._global_shift == 0 and isinstance(result, controlled_gate.ControlledGate) and isinstance(result.control_values, cv.ProductOfSums) and (result.control_values[-1] == (1,)) and (result.control_qid_shape[-1] == 2):
            return cirq.CCZPowGate(exponent=self._exponent, global_shift=self._global_shift).controlled(result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1])
        return result

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=('@', '@'), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None
        args.validate_version('2.0')
        return args.format('cz {0},{1};\n', qubits[0], qubits[1])

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CZ'
        return f'CZ**{self._exponent!r}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CZ'
            return f'(cirq.CZ**{proper_repr(self._exponent)})'
        return f'cirq.CZPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'