import collections
import copy
import logging
import math
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
def _map_free_gates(state, gates):
    """Map all gates that can be executed with the current layout.

    Args:
        state (_SystemState): The physical characteristics of the system, including its current
            layout and the coupling map.
        gates (list): Gates to be mapped.

    Returns:
        tuple:
            mapped_gates (list): ops for gates that can be executed, mapped onto layout.
            remaining_gates (list): gates that cannot be executed on the layout.
    """
    blocked_qubits = set()
    mapped_gates = []
    remaining_gates = []
    layout_map = state.layout._v2p
    for gate in gates:
        if not gate['partition']:
            qubits = _first_op_node(gate['graph']).qargs
            if not qubits:
                continue
            if blocked_qubits.intersection(qubits):
                blocked_qubits.update(qubits)
                remaining_gates.append(gate)
            else:
                mapped_gate = _transform_gate_for_system(gate, state)
                mapped_gates.append(mapped_gate)
            continue
        qubits = gate['partition'][0]
        if blocked_qubits.intersection(qubits):
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)
        elif len(qubits) == 1:
            mapped_gate = _transform_gate_for_system(gate, state)
            mapped_gates.append(mapped_gate)
        elif state.coupling_map.distance(layout_map[qubits[0]], layout_map[qubits[1]]) == 1:
            mapped_gate = _transform_gate_for_system(gate, state)
            mapped_gates.append(mapped_gate)
        else:
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)
    return (mapped_gates, remaining_gates)