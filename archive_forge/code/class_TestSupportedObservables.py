import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
class TestSupportedObservables:
    """Test that the device can implement all observables that it supports."""

    @pytest.mark.parametrize('observable', all_obs)
    def test_supported_observables_can_be_implemented(self, device_kwargs, observable):
        """Test that the device can implement all its supported observables."""
        device_kwargs['wires'] = 3
        dev = qml.device(**device_kwargs)
        if dev.shots and observable == 'SparseHamiltonian':
            pytest.skip('SparseHamiltonian only supported in analytic mode')
        if isinstance(dev, qml.Device):
            assert hasattr(dev, 'observables')
            if observable not in dev.observables:
                pytest.skip('observable not supported')
        kwargs = {'diff_method': 'parameter-shift'} if observable == 'SparseHamiltonian' else {}

        @qml.qnode(dev, **kwargs)
        def circuit(obs_circ):
            qml.PauliX(0)
            return qml.expval(obs_circ)
        if observable == 'Projector':
            for o in obs[observable]:
                assert isinstance(circuit(o), (float, np.ndarray))
        else:
            assert isinstance(circuit(obs[observable]), (float, np.ndarray))

    def test_tensor_observables_can_be_implemented(self, device_kwargs):
        """Test that the device can implement a simple tensor observable.
        This test is skipped for devices that do not support tensor observables."""
        device_kwargs['wires'] = 2
        dev = qml.device(**device_kwargs)
        supports_tensor = isinstance(dev, qml.devices.Device) or ('supports_tensor_observables' in dev.capabilities() and dev.capabilities()['supports_tensor_observables'])
        if not supports_tensor:
            pytest.skip('Device does not support tensor observables.')

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))
        assert isinstance(circuit(), (float, np.ndarray))