from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_qubit_pair_self_loops() -> v2.device_pb2.DeviceSpecification:
    """Creates an invalid DeviceSpecification with a qubit pair ('0_0', '0_0')."""
    q_proto_id = v2.qubit_to_proto_id(cirq.GridQubit(0, 0))
    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id])
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = targets.targets.add()
    new_target.ids.extend([q_proto_id, q_proto_id])
    return spec