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
def _score_step(step):
    """Count the mapped two-qubit gates, less the number of added SWAPs."""
    return len([g for g in step.gates_mapped if len(g.qargs) == 2]) - 3 * len(step.swaps_added)