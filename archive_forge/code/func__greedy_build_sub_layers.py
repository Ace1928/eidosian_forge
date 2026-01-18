from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
@staticmethod
def _greedy_build_sub_layers(current_layer: dict[tuple[int, int], Gate]) -> list[dict[tuple[int, int], Gate]]:
    """The greedy method of building sub-layers of commuting gates."""
    sub_layers = []
    while len(current_layer) > 0:
        current_sub_layer, remaining_gates = ({}, {})
        blocked_vertices: set[tuple] = set()
        for edge, evo_gate in current_layer.items():
            if blocked_vertices.isdisjoint(edge):
                current_sub_layer[edge] = evo_gate
                blocked_vertices = blocked_vertices.union(edge)
            else:
                remaining_gates[edge] = evo_gate
        current_layer = remaining_gates
        sub_layers.append(current_sub_layer)
    return sub_layers