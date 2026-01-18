from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
@value.value_equality()
class CSwapGate(gate_features.InterchangeableQubitsGate, raw_types.Gate):
    """A controlled swap gate. The Fredkin gate."""

    def qubit_index_to_equivalence_group_key(self, index):
        return 0 if index == 0 else 1

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return value.LinearDict({'III': 3 / 4, 'IXX': 1 / 4, 'IYY': 1 / 4, 'IZZ': 1 / 4, 'ZII': 1 / 4, 'ZXX': -1 / 4, 'ZYY': -1 / 4, 'ZZZ': -1 / 4})

    def _trace_distance_bound_(self) -> float:
        return 1.0

    def _decompose_(self, qubits):
        c, t1, t2 = qubits
        if hasattr(t1, 'is_adjacent'):
            if not t1.is_adjacent(t2):
                return self._decompose_inside_control(t1, c, t2)
            if not t1.is_adjacent(c):
                return self._decompose_outside_control(c, t2, t1)
        return self._decompose_outside_control(c, t1, t2)

    def _decompose_inside_control(self, target1: 'cirq.Qid', control: 'cirq.Qid', target2: 'cirq.Qid') -> 'cirq.OP_TREE':
        """A decomposition assuming the control separates the targets.

        target1: ─@─X───────T──────@────────@─────────X───@─────X^-0.5─
                  │ │              │        │         │   │
        control: ─X─@─X─────@─T^-1─X─@─T────X─@─X^0.5─@─@─X─@──────────
                      │     │        │        │         │   │
        target2: ─────@─H─T─X─T──────X─T^-1───X─T^-1────X───X─H─S^-1───
        """
        a, b, c = (target1, control, target2)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, a)
        yield common_gates.CNOT(c, b)
        yield common_gates.H(c)
        yield common_gates.T(c)
        yield common_gates.CNOT(b, c)
        yield common_gates.T(a)
        yield (common_gates.T(b) ** (-1))
        yield common_gates.T(c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.T(b)
        yield (common_gates.T(c) ** (-1))
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield (pauli_gates.X(b) ** 0.5)
        yield (common_gates.T(c) ** (-1))
        yield common_gates.CNOT(b, a)
        yield common_gates.CNOT(b, c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.H(c)
        yield (common_gates.S(c) ** (-1))
        yield (pauli_gates.X(a) ** (-0.5))

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        return protocols.apply_unitary(controlled_gate.ControlledGate(swap_gates.SWAP), protocols.ApplyUnitaryArgs(args.target_tensor, args.available_buffer, args.axes), default=NotImplemented)

    def _decompose_outside_control(self, control: 'cirq.Qid', near_target: 'cirq.Qid', far_target: 'cirq.Qid') -> 'cirq.OP_TREE':
        """A decomposition assuming one of the targets is in the middle.

        control: ───T──────@────────@───@────────────@────────────────
                           │        │   │            │
           near: ─X─T──────X─@─T^-1─X─@─X────@─X^0.5─X─@─X^0.5────────
                  │          │        │      │         │
            far: ─@─Y^-0.5─T─X─T──────X─T^-1─X─T^-1────X─S─────X^-0.5─
        """
        a, b, c = (control, near_target, far_target)
        t = common_gates.T
        sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]
        yield common_gates.CNOT(c, b)
        yield (pauli_gates.Y(c) ** (-0.5))
        yield (t(a), t(b), t(c))
        yield sweep_abc
        yield (t(b) ** (-1), t(c))
        yield sweep_abc
        yield (t(c) ** (-1))
        yield sweep_abc
        yield (t(c) ** (-1))
        yield (pauli_gates.X(b) ** 0.5)
        yield sweep_abc
        yield common_gates.S(c)
        yield (pauli_gates.X(b) ** 0.5)
        yield (pauli_gates.X(c) ** (-0.5))

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return linalg.block_diag(np.diag([1, 1, 1, 1, 1]), np.array([[0, 1], [1, 0]]), np.diag([1]))

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        if not args.use_unicode_characters:
            return protocols.CircuitDiagramInfo(('@', 'swap', 'swap'))
        return protocols.CircuitDiagramInfo(('@', '×', '×'))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('cswap {0},{1},{2};\n', qubits[0], qubits[1], qubits[2])

    def _value_equality_values_(self):
        return ()

    def __pow__(self, power):
        if power == 1 or power == -1:
            return self
        return NotImplemented

    def __str__(self) -> str:
        return 'FREDKIN'

    def __repr__(self) -> str:
        return 'cirq.FREDKIN'

    def _num_qubits_(self) -> int:
        return 3

    def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> raw_types.Gate:
        """Returns a controlled `SWAP` with one additional control.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = SWAP`.
        """
        if num_controls == 0:
            return self
        return controlled_gate.ControlledGate(controlled_gate.ControlledGate(swap_gates.SWAP, num_controls=1), num_controls=num_controls, control_values=control_values, control_qid_shape=control_qid_shape)