from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_with_all_couplings():
    _, spec = _create_device_spec_with_horizontal_couplings()
    for row in range(GRID_HEIGHT - 1):
        for col in range(2):
            new_target = spec.valid_targets[0].targets.add()
            new_target.ids.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, col)), v2.qubit_to_proto_id(cirq.GridQubit(row + 1, col))])
    return spec