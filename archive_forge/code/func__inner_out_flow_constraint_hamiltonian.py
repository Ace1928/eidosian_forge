import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def _inner_out_flow_constraint_hamiltonian(graph: Union[nx.DiGraph, rx.PyDiGraph], node: int) -> Hamiltonian:
    """Calculates the inner portion of the Hamiltonian in :func:`out_flow_constraint`.
    For a given :math:`i`, this function returns:

    .. math::

        d_{i}^{out}(d_{i}^{out} - 2)\\mathbb{I}
        - 2(d_{i}^{out}-1)\\sum_{j,(i,j)\\in E}\\hat{Z}_{ij} +
        ( \\sum_{j,(i,j)\\in E}\\hat{Z}_{ij} )^{2}

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        node: a fixed node

    Returns:
        qml.Hamiltonian: The inner part of the out-flow constraint Hamiltonian.
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}')
    coeffs = []
    ops = []
    is_rx = isinstance(graph, rx.PyDiGraph)
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T
    edges_to_qubits = edges_to_wires(graph)
    out_edges = graph.out_edges(node)
    d = len(out_edges)
    if is_rx:
        out_edges = sorted(out_edges)
    for edge in out_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wire = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(1)
        ops.append(qml.Z(wire))
    coeffs, ops = _square_hamiltonian_terms(coeffs, ops)
    for edge in out_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wire = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(-2 * (d - 1))
        ops.append(qml.Z(wire))
    coeffs.append(d * (d - 2))
    ops.append(qml.Identity(0))
    H = Hamiltonian(coeffs, ops)
    H.simplify()
    H.grouping_indices = [list(range(len(H.ops)))]
    return H