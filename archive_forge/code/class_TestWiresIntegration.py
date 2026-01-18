import pytest
import pennylane as qml
from pennylane import numpy as np
class TestWiresIntegration:
    """Test that the device integrates with PennyLane's wire management."""

    @pytest.mark.parametrize('wires1, wires2', [(['a', 'c', 'd'], [2, 3, 0]), ([-1, -2, -3], ['q1', 'ancilla', 2]), (['a', 'c'], [3, 0]), ([-1, -2], ['ancilla', 2]), (['a'], ['nothing'])])
    @pytest.mark.parametrize('circuit_factory', [make_simple_circuit_expval])
    def test_wires_expval(self, device, circuit_factory, wires1, wires2, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = device(wires1)
        dev2 = device(wires2)
        circuit1 = circuit_factory(dev1, wires1)
        circuit2 = circuit_factory(dev2, wires2)
        assert np.allclose(circuit1(), circuit2(), atol=tol(dev1.shots))