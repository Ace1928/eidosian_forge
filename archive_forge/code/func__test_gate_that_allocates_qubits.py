from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
def _test_gate_that_allocates_qubits(gate):
    from cirq.protocols.unitary_protocol import _strat_unitary_from_decompose
    op = gate.on(*cirq.LineQubit.range(cirq.num_qubits(gate)))
    moment = cirq.Moment(op)
    circuit = cirq.FrozenCircuit(op)
    circuit_op = cirq.CircuitOperation(circuit)
    for val in [gate, op, moment, circuit, circuit_op]:
        unitary_from_strat = _strat_unitary_from_decompose(val)
        assert unitary_from_strat is not None
        np.testing.assert_allclose(unitary_from_strat, gate.narrow_unitary())