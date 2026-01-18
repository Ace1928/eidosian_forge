import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def _set_default_node(self, key):
    """Create a new node if key not found"""
    if key not in self._key_to_node_index:
        self._key_to_node_index[key] = self._graph.add_node(NodeData(key=key, equivs=[]))
    return self._key_to_node_index[key]