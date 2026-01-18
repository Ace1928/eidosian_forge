import json
from typing import cast, List, Optional, Union, Type
import pathlib
import time
import google.protobuf.text_format as text_format
import cirq
from cirq.sim.simulator import SimulatesSamples
from cirq_google.api import v2
from cirq_google.engine import calibration, engine_validator, simulated_local_processor, util
from cirq_google.devices import grid_device
from cirq_google.devices.google_noise_properties import NoiseModelFromGoogleNoiseProperties
from cirq_google.engine.calibration_to_noise_properties import noise_properties_from_calibration
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
def _create_device_spec_from_template(template_name: str) -> v2.device_pb2.DeviceSpecification:
    """Load a template proto into a `v2.device_pb2.DeviceSpecification`."""
    path = pathlib.Path(__file__).parent.parent.resolve()
    with path.joinpath('devices', 'specifications', template_name).open() as f:
        proto_txt = f.read()
    device_spec = v2.device_pb2.DeviceSpecification()
    text_format.Parse(proto_txt, device_spec)
    return device_spec