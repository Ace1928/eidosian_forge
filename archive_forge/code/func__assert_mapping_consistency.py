import itertools
from typing import (
import numpy as np
import networkx as nx
from cirq import circuits, ops, value
import cirq.contrib.acquaintance as cca
from cirq.contrib import circuitdag
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph
def _assert_mapping_consistency(self):
    assert sorted(self._log_to_phys) == sorted(self.logical_qubits)
    assert sorted(self._phys_to_log) == sorted(self.physical_qubits)
    for l in self._log_to_phys:
        assert l == self._phys_to_log[self._log_to_phys[l]]