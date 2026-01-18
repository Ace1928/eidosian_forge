import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@flaky(max_runs=10)
class TestHamiltonianSupport:
    """Separate test to ensure that the device can differentiate Hamiltonian observables."""

    def test_hamiltonian_diff(self, device_kwargs, tol):
        """Tests a simple VQE gradient using parameter-shift rules."""
        device_kwargs['wires'] = 1
        dev = qml.device(**device_kwargs)
        coeffs = np.array([-0.05, 0.17])
        param = np.array(1.7, requires_grad=True)

        @qml.qnode(dev, diff_method='parameter-shift')
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.Hamiltonian(coeffs, [qml.X(0), qml.Z(0)]))
        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        def circuit1(param):
            """First Pauli subcircuit"""
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.X(0))

        def circuit2(param):
            """Second Pauli subcircuit"""
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.Z(0))
        half1 = qml.QNode(circuit1, dev, diff_method='parameter-shift')
        half2 = qml.QNode(circuit2, dev, diff_method='parameter-shift')

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)
        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)
        assert np.allclose(grad[0], grad_expected[0], atol=tol(dev.shots))
        assert np.allclose(grad[1], grad_expected[1], atol=tol(dev.shots))