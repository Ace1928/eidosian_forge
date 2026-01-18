from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_duplicate_qubit() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification with a qubit name that does not conform to '<int>_<int>'."""
    q_proto_id = v2.qubit_to_proto_id(cirq.GridQubit(0, 0))
    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id, q_proto_id])
    return spec