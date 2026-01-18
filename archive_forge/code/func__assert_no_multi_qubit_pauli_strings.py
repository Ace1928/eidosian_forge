import cirq
from cirq.contrib.paulistring import (
def _assert_no_multi_qubit_pauli_strings(circuit: cirq.Circuit) -> None:
    for op in circuit.all_operations():
        if isinstance(op, cirq.PauliStringGateOperation):
            assert len(op.pauli_string) == 1