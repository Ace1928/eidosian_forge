import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def _raise_if_shape_mismatch(gate, circuit):
    if gate.num_qubits != circuit.num_qubits or gate.num_clbits != circuit.num_clbits:
        raise CircuitError('Cannot add equivalence between circuit and gate of different shapes. Gate: {} qubits and {} clbits. Circuit: {} qubits and {} clbits.'.format(gate.num_qubits, gate.num_clbits, circuit.num_qubits, circuit.num_clbits))