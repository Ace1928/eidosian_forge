import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def _get_quantum_executables():
    qubits = cirq.GridQubit.rect(1, 5, 5, 0)
    return [QuantumExecutable(spec=_get_example_spec(name=f'example-program-{i}'), problem_topology=cirq.LineTopology(5), circuit=_get_random_circuit(qubits, random_state=i), measurement=BitstringsMeasurement(n_repetitions=10)) for i in range(3)]