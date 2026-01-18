import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@flaky(max_runs=10)
class TestTensorVar:
    """Test tensor variance measurements."""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'supports_tensor_observables': False})
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.X(0) @ qml.Y(2))
        res = circuit()
        expected = (8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2 - np.cos(2 * (theta - phi)) - np.cos(2 * (theta + phi)) + 2 * np.cos(2 * theta) + 2 * np.cos(2 * phi) + 14) / 16
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_pauliz_hadamard(self, device, tol, skip_if):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'supports_tensor_observables': False})
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.Z(0) @ qml.Hadamard(wires=[1]) @ qml.Y(2))
        res = circuit()
        expected = (3 + np.cos(2 * phi) * np.cos(varphi) ** 2 - np.cos(2 * theta) * np.sin(varphi) ** 2 - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)) / 4
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('base_obs, permuted_obs', list(zip(obs_lst, obs_permuted_lst)))
    def test_wire_order_in_tensor_prod_observables(self, device, base_obs, permuted_obs, tol, skip_if):
        """Test that when given a tensor observable the variance is the same regardless of the order of terms
        in the tensor observable, provided the wires each term acts on remain constant.

        eg:
        ob1 = qml.Z(0) @ qml.Y(1)
        ob2 = qml.Y(1) @ qml.Z(0)

        @qml.qnode(dev)
        def circ(obs):
            return qml.var(obs)

        circ(ob1) == circ(ob2)
        """
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'supports_tensor_observables': False})

        @qml.qnode(dev)
        def circ(ob):
            sub_routine(label_map=range(3))
            return qml.var(ob)
        assert np.allclose(circ(base_obs), circ(permuted_obs), atol=tol(dev.shots), rtol=0)

    @pytest.mark.parametrize('label_map', label_maps)
    def test_wire_label_in_tensor_prod_observables(self, device, label_map, tol, skip_if):
        """Test that when given a tensor observable the variance is the same regardless of how the
        wires are labelled, as long as they match the device order.

        eg:
        dev1 = qml.device("default.qubit", wires=[0, 1, 2])
        dev2 = qml.device("default.qubit", wires=['c', 'b', 'a']

        def circ(wire_labels):
            return qml.var(qml.Z(wire_labels[0]) @ qml.X(wire_labels[2]))

        c1, c2 = qml.QNode(circ, dev1), qml.QNode(circ, dev2)
        c1([0, 1, 2]) == c2(['c', 'b', 'a'])
        """
        dev = device(wires=3)
        dev_custom_labels = device(wires=label_map)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'supports_tensor_observables': False})

        def circ(wire_labels):
            sub_routine(wire_labels)
            return qml.var(qml.X(wire_labels[0]) @ qml.Y(wire_labels[1]) @ qml.Z(wire_labels[2]))
        circ_base_label = qml.QNode(circ, device=dev)
        circ_custom_label = qml.QNode(circ, device=dev_custom_labels)
        assert np.allclose(circ_base_label(wire_labels=range(3)), circ_custom_label(wire_labels=label_map), atol=tol(dev.shots), rtol=0)

    def test_hermitian(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            if 'Hermitian' not in dev.observables:
                pytest.skip('Skipped because device does not support the Hermitian observable.')
            skip_if(dev, {'supports_tensor_observables': False})
        theta = 0.432
        phi = 0.123
        varphi = -0.543
        A_ = 0.1 * np.array([[-6, 2 + 1j, -3, -5 + 2j], [2 - 1j, 0, 2 - 1j, -5 + 4j], [-3, 2 + 1j, 0, -4 + 3j], [-5 - 2j, -5 - 4j, -4 - 3j, -6]])

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.Z(0) @ qml.Hermitian(A_, wires=[1, 2]))
        res = circuit()
        expected = 0.01 * (1057 - np.cos(2 * phi) + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi) - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi)) + 16 * np.sin(2 * phi) - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi) - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2 - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi) - 8 * np.cos(theta) * (4 * np.cos(phi) * (4 + 8 * np.cos(varphi) + np.cos(2 * varphi) - (1 + 6 * np.cos(varphi)) * np.sin(varphi)) + np.sin(phi) * (15 + 8 * np.cos(varphi) - 11 * np.cos(2 * varphi) + 42 * np.sin(varphi) + 3 * np.sin(2 * varphi)))) / 16
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_projector(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Projector works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            if 'Projector' not in dev.observables:
                pytest.skip('Skipped because device does not support the Projector observable.')
            skip_if(dev, {'supports_tensor_observables': False})
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.Z(0) @ qml.Projector(basis_state, wires=[1, 2]))
        res_basis = circuit([0, 0])
        res_state = circuit([1, 0, 0, 0])
        expected = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))
        res_basis = circuit([0, 1])
        res_state = circuit([0, 1, 0, 0])
        expected = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))
        res_basis = circuit([1, 0])
        res_state = circuit([0, 0, 1, 0])
        expected = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))
        res_basis = circuit([1, 1])
        res_state = circuit([0, 0, 0, 1])
        expected = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))
        res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
        expected_mean = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 - (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 - (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2)
        expected_var = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2) - expected_mean ** 2
        assert np.allclose(res, expected_var, atol=tol(False))