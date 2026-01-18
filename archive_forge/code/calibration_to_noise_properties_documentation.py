from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING
from cirq import ops
from cirq.devices import noise_utils
from cirq_google import engine
from cirq_google import ops as cg_ops
from cirq_google.devices import google_noise_properties
from cirq_google.engine import util
Translates between `cirq_google.Calibration` and NoiseProperties.

    The NoiseProperties object can then be used as input to the
    `cirq_google.NoiseModelFromGoogleNoiseProperties` class to create a
    `cirq.NoiseModel` that can be used with a simulator.

    To manually override noise properties, call `with_params` on the output:

        >>> cal = cirq_google.engine.load_median_device_calibration("rainbow")
        >>> # noise_props with all gate durations set to 37ns.
        >>> noise_props = cirq_google.engine.noise_properties_from_calibration(cal).with_params(
        ...     gate_times_ns=37)

    See `cirq_google.GoogleNoiseProperties` for details.

    Args:
        calibration: a Calibration object with hardware metrics.
        zphase_data: Optional data for Z phases not captured by Calibration -
            specifically, zeta and gamma. These values require Floquet
            calibration and can be provided here if available.
        gate_times_ns: Map of gate durations in nanoseconds. If not provided,
            defaults to the Sycamore gate times listed in `known_devices.py`.

    Returns:
        A `cirq_google.GoogleNoiseProperties` which represents the error
        present in the given Calibration object.
    