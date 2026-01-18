import re
from collections import OrderedDict
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit, ControlFlowOp
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, PowerModifier
from qiskit.circuit.tools import pi_check
from qiskit.converters import circuit_to_dag
from qiskit.utils import optionals as _optionals
from ..exceptions import VisualizationError
def _any_crossover(qubits, node, nodes):
    """Return True .IFF. 'node' crosses over any 'nodes'."""
    gate_span = _get_gate_span(qubits, node)
    all_indices = []
    for check_node in nodes:
        if check_node != node:
            all_indices += _get_gate_span(qubits, check_node)
    return any((i in gate_span for i in all_indices))