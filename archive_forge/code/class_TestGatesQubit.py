from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
@flaky(max_runs=10)
class TestGatesQubit:
    """Test qubit-based devices' probability vector after application of gates."""

    @pytest.mark.parametrize('basis_state', [np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([1, 1, 1, 1])])
    def test_basis_state(self, device, basis_state, tol, skip_if):
        """Test basis state initialization."""
        n_wires = 4
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(basis_state, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))
        res = circuit()
        expected = np.zeros([2 ** n_wires])
        expected[np.ravel_multi_index(basis_state, [2] * n_wires)] = 1
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_state_prep(self, device, init_state, tol, skip_if):
        """Test StatePrep initialisation."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            return qml.probs(range(n_wires))
        res = circuit()
        expected = np.abs(rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('op,mat', single_qubit)
    def test_single_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
        """Test PauliX application."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('gamma', [0.5432, -0.232])
    @pytest.mark.parametrize('op,func', single_qubit_param)
    def test_single_qubit_parameters(self, device, init_state, op, func, gamma, tol, skip_if, benchmark):
        """Test single qubit gates taking a single scalar argument."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            op(gamma, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(func(gamma) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_rotation(self, device, init_state, tol, skip_if, benchmark):
        """Test three axis rotation gate."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)
        a = 0.542
        b = 1.3432
        c = -0.654

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            qml.Rot(a, b, c, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(rot(a, b, c) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('op,mat', two_qubit)
    def test_two_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
        """Test two qubit gates."""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
            if not dev.supports_operation(op(wires=range(n_wires)).name):
                pytest.skip('op not supported')
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('param', [0.5432, -0.232])
    @pytest.mark.parametrize('op,func', two_qubit_param)
    def test_two_qubit_parameters(self, device, init_state, op, func, param, tol, skip_if, benchmark):
        """Test parametrized two qubit gates taking a single scalar argument."""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            op(param, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(func(param) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('mat', [U, U2])
    def test_qubit_unitary(self, device, init_state, mat, tol, skip_if, benchmark):
        """Test QubitUnitary gate."""
        n_wires = int(np.log2(len(mat)))
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            if 'QubitUnitary' not in dev.operations:
                pytest.skip('Skipped because device does not support QubitUnitary.')
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            qml.QubitUnitary(mat, wires=list(range(n_wires)))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('theta_', [np.array([0.4, -0.1, 0.2]), np.ones(15) / 3])
    def test_special_unitary(self, device, init_state, theta_, tol, skip_if, benchmark):
        """Test SpecialUnitary gate."""
        n_wires = int(np.log(len(theta_) + 1) / np.log(4))
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            if 'SpecialUnitary' not in dev.operations:
                pytest.skip('Skipped because device does not support SpecialUnitary.')
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            qml.SpecialUnitary(theta_, wires=list(range(n_wires)))
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        basis_fn = qml.ops.qubit.special_unitary.pauli_basis_matrices
        basis = basis_fn(n_wires)
        mat = qml.math.expm(1j * np.tensordot(theta_, basis, axes=[[0], [0]]))
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize('op, mat', three_qubit)
    def test_three_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
        """Test three qubit gates without parameters."""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
            skip_if(dev, {'returns_probs': False})
        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(rnd_state, wires=range(n_wires))
            op(wires=[0, 1, 2])
            return qml.probs(wires=range(n_wires))
        res = benchmark(circuit)
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))