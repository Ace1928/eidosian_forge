import pytest
import pennylane.numpy as pnp
import pennylane as qml
class TestDeviceProperties:
    """Test the device is created with the expected properties."""

    def test_load_device(self, device_kwargs):
        """Test that the device loads correctly."""
        device_kwargs['wires'] = 2
        device_kwargs['shots'] = 1234
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            assert isinstance(dev.wires, qml.wires.Wires)
            assert dev.wires == qml.wires.Wires((0, 1))
            assert isinstance(dev.shots, qml.measurements.Shots)
            assert dev.shots == qml.measurements.Shots(1234)
            assert device_kwargs['name'] == dev.name
            assert isinstance(dev.tracker, qml.Tracker)
            return
        assert dev.num_wires == 2
        assert dev.shots == 1234
        assert dev.short_name == device_kwargs['name']

    def test_no_wires_given(self, device_kwargs):
        """Test that the device requires correct arguments."""
        with pytest.raises(TypeError, match='missing 1 required positional argument'):
            dev = qml.device(**device_kwargs)
            if isinstance(dev, qml.devices.Device):
                pytest.skip('test is old interface specific.')

    def test_no_0_shots(self, device_kwargs):
        """Test that non-analytic devices cannot accept 0 shots."""
        device_kwargs['wires'] = 2
        device_kwargs['shots'] = 0
        with pytest.raises(Exception):
            dev = qml.device(**device_kwargs)
            if isinstance(dev, qml.devices.Device):
                pytest.skip('test is old interface specific.')