from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _check_edges(self, dag: DAGCircuit, node: DAGOpNode, swap_strategy: SwapStrategy):
    """Check if the swap strategy can create the required connectivity.

        Args:
            node: The dag node for which to check if the swap strategy provides enough connectivity.
            swap_strategy: The swap strategy that is being used.

        Raises:
            TranspilerError: If there is an edge that the swap strategy cannot accommodate
                and if the pass has been configured to raise on such issues.
        """
    required_edges = set()
    for sub_node in node.op:
        edge = (dag.find_bit(sub_node.qargs[0]).index, dag.find_bit(sub_node.qargs[1]).index)
        required_edges.add(edge)
    if not required_edges.issubset(swap_strategy.possible_edges):
        raise TranspilerError(f'{swap_strategy} cannot implement all edges in {required_edges}.')