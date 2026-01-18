import itertools
from typing import Iterable
import cirq
import networkx as nx
def _Device_dot_get_nx_graph(device: 'cirq.Device') -> nx.Graph:
    """Shim over future `cirq.Device` method to get a NetworkX graph."""
    if device.metadata is not None:
        return device.metadata.nx_graph
    raise ValueError('Supplied device must contain metadata.')