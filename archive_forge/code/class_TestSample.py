import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@flaky(max_runs=10)
class TestSample:
    """Tests for the sample return type."""

    def test_sample_values(self, device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.5708, wires=[0])
            return qml.sample(qml.Z(0))
        res = circuit()
        assert np.allclose(res ** 2, 1, atol=tol(False))

    def test_sample_values_hermitian(self, device, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
        if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
            pytest.skip('Skipped because device does not support the Hermitian observable.')
        A_ = np.array([[1, 2j], [-2j, 0]])
        theta = 0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            return qml.sample(qml.Hermitian(A_, wires=0))
        res = circuit().flatten()
        eigvals = np.linalg.eigvalsh(A_)
        assert np.allclose(sorted(list(set(res.tolist()))), sorted(eigvals), atol=tol(dev.shots))
        assert np.allclose(np.mean(res), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, atol=tol(False))
        assert np.allclose(np.var(res), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, atol=tol(False))

    def test_sample_values_projector(self, device, tol):
        """Tests if the samples of a Projector observable returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
        if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
            pytest.skip('Skipped because device does not support the Projector observable.')
        theta = 0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            return qml.sample(qml.Projector(state, wires=0))
        expected = np.cos(theta / 2) ** 2
        res_basis = circuit([0]).flatten()
        res_state = circuit([1, 0]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected - expected ** 2, atol=tol(False))
        assert np.allclose(np.var(res_state), expected - expected ** 2, atol=tol(False))
        expected = np.sin(theta / 2) ** 2
        res_basis = circuit([1]).flatten()
        res_state = circuit([0, 1]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected - expected ** 2, atol=tol(False))
        assert np.allclose(np.var(res_state), expected - expected ** 2, atol=tol(False))
        expected = 0.5
        res = circuit(np.array([1, 1]) / np.sqrt(2)).flatten()
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res), expected, atol=tol(False))
        assert np.allclose(np.var(res), expected - expected ** 2, atol=tol(False))

    def test_sample_values_hermitian_multi_qubit(self, device, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
        if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
            pytest.skip('Skipped because device does not support the Hermitian observable.')
        theta = 0.543
        A_ = np.array([[1, 2j, 1 - 2j, 0.5j], [-2j, 0, 3 + 4j, 1], [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j], [-0.5j, 1, 1.5 + 2j, -1]])

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RY(2 * theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Hermitian(A_, wires=[0, 1]))
        res = circuit().flatten()
        eigvals = np.linalg.eigvalsh(A_)
        assert np.allclose(sorted(list(set(res.tolist()))), sorted(eigvals), atol=tol(dev.shots))
        expected = (88 * np.sin(theta) + 24 * np.sin(2 * theta) - 40 * np.sin(3 * theta) + 5 * np.cos(theta) - 6 * np.cos(2 * theta) + 27 * np.cos(3 * theta) + 6) / 32
        assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))

    def test_sample_values_projector_multi_qubit(self, device, tol):
        """Tests if the samples of a multi-qubit Projector observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
        if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
            pytest.skip('Skipped because device does not support the Projector observable.')
        theta = 0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            qml.RY(2 * theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Projector(state, wires=[0, 1]))
        expected = (np.cos(theta / 2) * np.cos(theta)) ** 2
        res_basis = circuit([0, 0]).flatten()
        res_state = circuit([1, 0, 0, 0]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))
        expected = (np.cos(theta / 2) * np.sin(theta)) ** 2
        res_basis = circuit([0, 1]).flatten()
        res_state = circuit([0, 1, 0, 0]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))
        expected = (np.sin(theta / 2) * np.sin(theta)) ** 2
        res_basis = circuit([1, 0]).flatten()
        res_state = circuit([0, 0, 1, 0]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))
        expected = (np.sin(theta / 2) * np.cos(theta)) ** 2
        res_basis = circuit([1, 1]).flatten()
        res_state = circuit([0, 0, 0, 1]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))