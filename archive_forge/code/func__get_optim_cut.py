import uuid
from typing import Any, Callable, Sequence, Tuple
import warnings
import numpy as np
from networkx import MultiDiGraph, has_path, weakly_connected_components
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.meta import WireCut
from pennylane.queuing import WrappedObj
from pennylane.operation import Operation
from .kahypar import kahypar_cut
from .cutstrategy import CutStrategy
def _get_optim_cut(valid_cut_edges, local_measurement=False):
    """Picks out the best cut from a dict of valid candidate cuts."""
    if local_measurement:
        min_max_node_degree = min((max_node_degree for _, max_node_degree in valid_cut_edges))
        optim_cuts = {k: cut_edges for (k, max_node_degree), cut_edges in valid_cut_edges.items() if max_node_degree == min_max_node_degree}
    else:
        min_cuts = min((len(cut_edges) for cut_edges in valid_cut_edges.values()))
        optim_cuts = {k: cut_edges for (k, _), cut_edges in valid_cut_edges.items() if len(cut_edges) == min_cuts}
    return optim_cuts[min(optim_cuts)]