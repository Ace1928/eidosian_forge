import time
import logging
from functools import singledispatchmethod
from itertools import zip_longest
from collections import defaultdict
import rustworkx
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key, NodeData
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
def _basis_search(equiv_lib, source_basis, target_basis):
    """Search for a set of transformations from source_basis to target_basis.

    Args:
        equiv_lib (EquivalenceLibrary): Source of valid translations
        source_basis (Set[Tuple[gate_name: str, gate_num_qubits: int]]): Starting basis.
        target_basis (Set[gate_name: str]): Target basis.

    Returns:
        Optional[List[Tuple[gate, equiv_params, equiv_circuit]]]: List of (gate,
            equiv_params, equiv_circuit) tuples tuples which, if applied in order
            will map from source_basis to target_basis. Returns None if no path
            was found.
    """
    logger.debug('Begining basis search from %s to %s.', source_basis, target_basis)
    source_basis = {(gate_name, gate_num_qubits) for gate_name, gate_num_qubits in source_basis if gate_name not in target_basis}
    if not source_basis:
        return []
    target_basis_keys = [key for key in equiv_lib.keys() if key.name in target_basis]
    graph = equiv_lib.graph
    vis = BasisSearchVisitor(graph, source_basis, target_basis_keys)
    dummy = graph.add_node(NodeData(key='key', equivs=[('dummy starting node', 0)]))
    try:
        graph.add_edges_from_no_data([(dummy, equiv_lib.node_index(key)) for key in target_basis_keys])
        rtn = None
        try:
            rustworkx.digraph_dijkstra_search(graph, [dummy], vis.edge_cost, vis)
        except StopIfBasisRewritable:
            rtn = vis.basis_transforms
            logger.debug('Transformation path:')
            for gate_name, gate_num_qubits, params, equiv in rtn:
                logger.debug('%s/%s => %s\n%s', gate_name, gate_num_qubits, params, equiv)
    finally:
        graph.remove_node(dummy)
    return rtn