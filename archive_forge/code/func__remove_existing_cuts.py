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
def _remove_existing_cuts(graph: MultiDiGraph) -> MultiDiGraph:
    """Removes all existing, manually or automatically placed, cuts from a circuit graph, be it
    ``WireCut``s or ``MeasureNode``-``PrepareNode`` pairs.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.

    Returns:
        (MultiDiGraph): Copy of the input graph with all its existing cuts removed.
    """
    uncut_graph = graph.copy()
    for node in list(graph.nodes):
        if isinstance(node.obj, WireCut):
            uncut_graph.remove_node(node)
        elif isinstance(node.obj, MeasureNode):
            for node1 in graph.neighbors(node):
                if isinstance(node1.obj, PrepareNode):
                    uncut_graph.remove_node(node)
                    uncut_graph.remove_node(node1)
    if len([n for n in uncut_graph.nodes if isinstance(n.obj, (MeasureNode, PrepareNode))]) > 0:
        warnings.warn('The circuit contains `MeasureNode` or `PrepareNode` operations that are not paired up correctly. Please check.', UserWarning)
    return uncut_graph