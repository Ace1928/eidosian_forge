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
def acts_on_nonadjacent_qubits(self, op: ops.Operation) -> bool:
    if len(op.qubits) == 1:
        return False
    return tuple(self.log_to_phys(*op.qubits)) not in self.device_graph.edges