import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def _build_basis_graph(self):
    graph = rx.PyDiGraph()
    node_map = {}
    for key in self._key_to_node_index:
        name, num_qubits = key
        equivalences = self._get_equivalences(key)
        basis = frozenset([f'{name}/{num_qubits}'])
        for params, decomp in equivalences:
            decomp_basis = frozenset((f'{name}/{num_qubits}' for name, num_qubits in {(instruction.operation.name, instruction.operation.num_qubits) for instruction in decomp.data}))
            if basis not in node_map:
                basis_node = graph.add_node({'basis': basis, 'label': str(set(basis))})
                node_map[basis] = basis_node
            if decomp_basis not in node_map:
                decomp_basis_node = graph.add_node({'basis': decomp_basis, 'label': str(set(decomp_basis))})
                node_map[decomp_basis] = decomp_basis_node
            label = '{}\n{}'.format(str(params), str(decomp) if num_qubits <= 5 else '...')
            graph.add_edge(node_map[basis], node_map[decomp_basis], {'label': label, 'fontname': 'Courier', 'fontsize': str(8)})
    return graph