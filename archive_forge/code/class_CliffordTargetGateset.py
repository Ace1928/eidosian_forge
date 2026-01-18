from typing import List, Union, Type, cast, TYPE_CHECKING
from enum import Enum
import numpy as np
from cirq import ops, transformers, protocols, linalg
from cirq.type_workarounds import NotImplementedType
class CliffordTargetGateset(transformers.TwoQubitCompilationTargetGateset):
    """Target gateset containing CZ + Meas + SingleQubitClifford / PauliStringPhasor gates."""

    class SingleQubitTarget(Enum):
        SINGLE_QUBIT_CLIFFORDS = 1
        PAULI_STRING_PHASORS_AND_CLIFFORDS = 2
        PAULI_STRING_PHASORS = 3

    def __init__(self, *, single_qubit_target: SingleQubitTarget=SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS, atol: float=1e-08):
        """Initializes CliffordTargetGateset

        Args:
            single_qubit_target: Specifies the decomposition strategy for single qubit gates.
                SINGLE_QUBIT_CLIFFORDS: Decompose all single qubit gates to
                    `cirq.SingleQubitCliffordGate`.
                PAULI_STRING_PHASORS_AND_CLIFFORDS: Accept both `cirq.SingleQubitCliffordGate` and
                    `cirq.PauliStringPhasorGate`; but decompose unknown gates into
                    `cirq.PauliStringPhasorGate`.
                PAULI_STRING_PHASORS: Decompose all single qubit gates to
                    `cirq.PauliStringPhasorGate`.
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        self.atol = atol
        self.single_qubit_target = single_qubit_target
        gates: List[Union['cirq.Gate', Type['cirq.Gate']]] = [ops.CZ, ops.MeasurementGate]
        if single_qubit_target in [self.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS, self.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS]:
            gates.append(ops.SingleQubitCliffordGate)
        if single_qubit_target in [self.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS, self.SingleQubitTarget.PAULI_STRING_PHASORS]:
            gates.append(ops.PauliStringPhasorGate)
        super().__init__(*gates)

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> Union[NotImplementedType, 'cirq.OP_TREE']:
        if not protocols.has_unitary(op):
            return NotImplemented
        mat = protocols.unitary(op)
        keep_clifford = self.single_qubit_target == self.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS
        return _matrix_to_clifford_op(mat, op.qubits[0], atol=self.atol) if self.single_qubit_target == self.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS else _matrix_to_pauli_string_phasors(mat, op.qubits[0], keep_clifford=keep_clifford, atol=self.atol)

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> Union[NotImplementedType, 'cirq.OP_TREE']:
        if not protocols.has_unitary(op):
            return NotImplemented
        return transformers.two_qubit_matrix_to_cz_operations(op.qubits[0], op.qubits[1], protocols.unitary(op), allow_partial_czs=False, atol=self.atol)

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""

        def rewriter(o: 'cirq.CircuitOperation'):
            result = self._decompose_single_qubit_operation(o, -1)
            return o.circuit.all_operations() if result is NotImplemented else result
        return [transformers.create_transformer_with_kwargs(transformers.merge_k_qubit_unitaries, k=1, rewriter=rewriter), transformers.drop_negligible_operations, transformers.drop_empty_moments]