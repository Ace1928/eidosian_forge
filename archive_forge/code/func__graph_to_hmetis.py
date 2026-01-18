from collections.abc import Sequence as SequenceType
from itertools import compress
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
import numpy as np
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.operation import Operation
def _graph_to_hmetis(graph: MultiDiGraph, hyperwire_weight: int=0, edge_weights: Sequence[int]=None) -> Tuple[List[int], List[int], List[Union[int, float]]]:
    """Converts a ``MultiDiGraph`` into the
    `hMETIS hypergraph input format <http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf>`__
    conforming to KaHyPar's calling signature.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Defaults to 0 which leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together.
        edge_weights (Sequence[int]): Weights for regular edges in the graph. Defaults to ``None``,
            which leads to unit-weighted edges.

    Returns:
        Tuple[List,List,List]: The 3 lists representing an (optionally weighted) hypergraph:
        - Flattened list of adjacent node indices.
        - List of starting indices for edges in the above adjacent-nodes-list.
        - Optional list of edge weights. ``None`` if ``hyperwire_weight`` is equal to 0.
    """
    nodes = list(graph.nodes)
    edges = graph.edges(data='wire')
    wires = {w for _, _, w in edges}
    adj_nodes = [nodes.index(v) for ops in graph.edges(keys=False) for v in ops]
    edge_splits = qml.math.cumsum([0] + [len(e) for e in graph.edges(keys=False)]).tolist()
    edge_weights = edge_weights if edge_weights is not None and len(edges) == len(edge_weights) else None
    if hyperwire_weight:
        hyperwires = {w: set() for w in wires}
        num_wires = len(hyperwires)
        for v0, v1, wire in edges:
            hyperwires[wire].update([nodes.index(v0), nodes.index(v1)])
        for wire, nodes_on_wire in hyperwires.items():
            nwv = len(nodes_on_wire)
            edge_splits.append(nwv + edge_splits[-1])
            adj_nodes = adj_nodes + list(nodes_on_wire)
        assert len(edge_splits) == len(edges) + num_wires + 1
        if isinstance(hyperwire_weight, (int, float)):
            edge_weights = edge_weights or [1] * len(edges)
            wire_weights = [hyperwire_weight] * num_wires
            edge_weights = edge_weights + wire_weights
    return (adj_nodes, edge_splits, edge_weights)