from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_unexpected_asymmetric_target() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification containing an ASYMMETRIC target set."""
    spec = v2.device_pb2.DeviceSpecification()
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.ASYMMETRIC
    return spec