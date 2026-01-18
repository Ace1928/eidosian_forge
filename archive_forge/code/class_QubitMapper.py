from typing import Callable
from cirq import circuits, devices, ops
class QubitMapper:

    def __init__(self, qubit_map: Callable[[ops.Qid], ops.Qid]) -> None:
        self.qubit_map = qubit_map

    def map_operation(self, operation: ops.Operation) -> ops.Operation:
        return operation.transform_qubits(self.qubit_map)

    def map_moment(self, moment: circuits.Moment) -> circuits.Moment:
        return circuits.Moment((self.map_operation(op) for op in moment.operations))

    def optimize_circuit(self, circuit: circuits.Circuit):
        circuit[:] = (self.map_moment(m) for m in circuit)