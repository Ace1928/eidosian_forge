import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def _raise_if_param_mismatch(gate_params, circuit_parameters):
    gate_parameters = [p for p in gate_params if isinstance(p, ParameterExpression)]
    if set(gate_parameters) != circuit_parameters:
        raise CircuitError('Cannot add equivalence between circuit and gate of different parameters. Gate params: {}. Circuit params: {}.'.format(gate_parameters, circuit_parameters))