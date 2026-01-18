from typing import List
import numpy as np
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
class AQTTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target gateset accepting XXPowGate + X/Y/Z/PhX single qubit rotations + measurement gates.

    By default, `cirq_aqt.AQTTargetGateset` will accept and compile unknown
    gates to the following universal target gateset:

    - `cirq.XXPowGate`: The two qubit entangling gate.
    - `cirq.XPowGate`, `cirq.YPowGate`, `cirq.ZPowGate`,
      `cirq.PhasedXPowGate`: Single qubit rotations.
    - `cirq.MeasurementGate`: Measurements.
    """

    def __init__(self):
        super().__init__(cirq.XXPowGate, cirq.MeasurementGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.PhasedXPowGate, unroll_circuit_op=False)

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _: int) -> DecomposeResult:
        opu = op.untagged
        opu = next(opu.circuit.all_operations()).untagged if isinstance(opu, cirq.CircuitOperation) and len(opu.circuit) == 1 else opu
        if isinstance(opu.gate, cirq.HPowGate) and opu.gate.exponent == 1:
            return [cirq.rx(np.pi).on(opu.qubits[0]), cirq.ry(-1 * np.pi / 2).on(opu.qubits[0])]
        if cirq.has_unitary(opu):
            gates = cirq.single_qubit_matrix_to_phased_x_z(cirq.unitary(opu))
            return [g.on(opu.qubits[0]) for g in gates]
        return NotImplemented

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if cirq.has_unitary(op):
            return cirq.two_qubit_matrix_to_ion_operations(op.qubits[0], op.qubits[1], cirq.unitary(op))
        return NotImplemented

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""
        return [cirq.drop_negligible_operations, cirq.drop_empty_moments]