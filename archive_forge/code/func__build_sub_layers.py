from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _build_sub_layers(self, current_layer: dict[tuple[int, int], Gate]) -> list[dict[tuple[int, int], Gate]]:
    """A helper method to build-up sets of gates to simultaneously apply.

        This is done with an edge coloring if the ``edge_coloring`` init argument was given or with
        a greedy algorithm if not. With an edge coloring all gates on edges with the same color
        will be applied simultaneously. These sublayers are applied in the order of their color,
        which is an int, in increasing color order.

        Args:
            current_layer: All gates in the current layer can be applied given the qubit ordering
                of the current layout. However, not all gates in the current layer can be applied
                simultaneously. This function creates sub-layers by building up sub-layers
                of gates. All gates in a sub-layer can simultaneously be applied given the coupling
                map and current qubit configuration.

        Returns:
             A list of gate dicts that can be applied. The gates a position 0 are applied first.
             A gate dict has the qubit tuple as key and the gate to apply as value.
        """
    if self._edge_coloring is not None:
        return self._edge_coloring_build_sub_layers(current_layer)
    else:
        return self._greedy_build_sub_layers(current_layer)