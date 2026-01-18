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
def _calc_layout_distance(gates, state, max_gates=None):
    """Return the sum of the distances of two-qubit pairs in each CNOT in gates
    according to the layout and the coupling.
    """
    if max_gates is None:
        max_gates = 50 + 10 * len(state.coupling_map.physical_qubits)
    layout_map = state.layout._v2p
    out = 0
    for gate in gates[:max_gates]:
        if not gate['partition']:
            continue
        qubits = gate['partition'][0]
        if len(qubits) == 2:
            out += state.coupling_map.distance(layout_map[qubits[0]], layout_map[qubits[1]])
    return out