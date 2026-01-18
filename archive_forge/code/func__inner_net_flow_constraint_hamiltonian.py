import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def _inner_net_flow_constraint_hamiltonian(graph: Union[nx.DiGraph, rx.PyDiGraph], node: int) -> Hamiltonian:
    """Calculates the squared inner portion of the Hamiltonian in :func:`net_flow_constraint`.


    For a given :math:`i`, this function returns:

    .. math::

        \\left((d_{i}^{\\rm out} - d_{i}^{\\rm in})\\mathbb{I} -
        \\sum_{j, (i, j) \\in E} Z_{ij} + \\sum_{j, (j, i) \\in E} Z_{ji} \\right)^{2}.

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        node: a fixed node

    Returns:
        qml.Hamiltonian: The inner part of the net-flow constraint Hamiltonian.
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}')
    edges_to_qubits = edges_to_wires(graph)
    coeffs = []
    ops = []
    is_rx = isinstance(graph, rx.PyDiGraph)
    out_edges = graph.out_edges(node)
    in_edges = graph.in_edges(node)
    if is_rx:
        out_edges = sorted(out_edges)
        in_edges = sorted(in_edges)
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T
    coeffs.append(len(out_edges) - len(in_edges))
    ops.append(qml.Identity(0))
    for edge in out_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wires = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(-1)
        ops.append(qml.Z(wires))
    for edge in in_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wires = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(1)
        ops.append(qml.Z(wires))
    coeffs, ops = _square_hamiltonian_terms(coeffs, ops)
    H = Hamiltonian(coeffs, ops)
    H.simplify()
    H.grouping_indices = [list(range(len(H.ops)))]
    return H