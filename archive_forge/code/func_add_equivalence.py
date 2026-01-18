import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def add_equivalence(self, gate, equivalent_circuit):
    """Add a new equivalence to the library. Future queries for the Gate
        will include the given circuit, in addition to all existing equivalences
        (including those from base).

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            equivalent_circuit (QuantumCircuit): A circuit equivalently
                implementing the given Gate.
        """
    _raise_if_shape_mismatch(gate, equivalent_circuit)
    _raise_if_param_mismatch(gate.params, equivalent_circuit.parameters)
    key = Key(name=gate.name, num_qubits=gate.num_qubits)
    equiv = Equivalence(params=gate.params.copy(), circuit=equivalent_circuit.copy())
    target = self._set_default_node(key)
    self._graph[target].equivs.append(equiv)
    sources = {Key(name=instruction.operation.name, num_qubits=len(instruction.qubits)) for instruction in equivalent_circuit}
    edges = [(self._set_default_node(source), target, EdgeData(index=self._rule_id, num_gates=len(sources), rule=equiv, source=source)) for source in sources]
    self._graph.add_edges_from(edges)
    self._rule_id += 1