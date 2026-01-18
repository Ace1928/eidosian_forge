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
def _get_gate_span(qubits, node):
    """Get the list of qubits drawing this gate would cover
    qiskit-terra #2802
    """
    min_index = len(qubits)
    max_index = 0
    for qreg in node.qargs:
        index = qubits.index(qreg)
        if index < min_index:
            min_index = index
        if index > max_index:
            max_index = index
    if isinstance(node.op, ControlFlowOp):
        span = qubits
    elif node.cargs or getattr(node.op, 'condition', None):
        span = qubits[min_index:len(qubits)]
    else:
        span = qubits[min_index:max_index + 1]
    return span