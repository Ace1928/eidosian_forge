import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def _partial_cycle_mixer(graph: Union[nx.DiGraph, rx.PyDiGraph], edge: Tuple) -> Hamiltonian:
    """Calculates the partial cycle-mixer Hamiltonian for a specific edge.

    For an edge :math:`(i, j)`, this function returns:

    .. math::

        \\sum_{k \\in V, k\\neq i, k\\neq j, (i, k) \\in E, (k, j) \\in E}\\left[
        X_{ij}X_{ik}X_{kj} + Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\\right]

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        edge (tuple): a fixed edge

    Returns:
        qml.Hamiltonian: the partial cycle-mixer Hamiltonian
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}')
    coeffs = []
    ops = []
    is_rx = isinstance(graph, rx.PyDiGraph)
    edges_to_qubits = edges_to_wires(graph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes
    graph_edges = sorted(graph.edge_list()) if is_rx else graph.edges
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T
    for node in graph_nodes:
        out_edge = (edge[0], node)
        in_edge = (node, edge[1])
        if node not in edge and out_edge in graph_edges and (in_edge in graph_edges):
            wire = edges_to_qubits[get_nvalues(edge)]
            out_wire = edges_to_qubits[get_nvalues(out_edge)]
            in_wire = edges_to_qubits[get_nvalues(in_edge)]
            t = qml.X(wire) @ qml.X(out_wire) @ qml.X(in_wire)
            ops.append(t)
            t = qml.Y(wire) @ qml.Y(out_wire) @ qml.X(in_wire)
            ops.append(t)
            t = qml.Y(wire) @ qml.X(out_wire) @ qml.Y(in_wire)
            ops.append(t)
            t = qml.X(wire) @ qml.Y(out_wire) @ qml.Y(in_wire)
            ops.append(t)
            coeffs.extend([0.25, 0.25, 0.25, -0.25])
    return Hamiltonian(coeffs, ops)