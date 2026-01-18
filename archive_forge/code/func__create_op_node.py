import math
import heapq
from collections import OrderedDict, defaultdict
import rustworkx as rx
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
def _create_op_node(self, operation, qargs, cargs):
    """Creates a DAGDepNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Operation): operation
            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to

        Returns:
            DAGDepNode: the newly added node.
        """
    directives = ['measure']
    if not getattr(operation, '_directive', False) and operation.name not in directives:
        qindices_list = []
        for elem in qargs:
            qindices_list.append(self.qubits.index(elem))
        if getattr(operation, 'condition', None):
            cond_bits = condition_resources(operation.condition).clbits
            cindices_list = [self.clbits.index(clbit) for clbit in cond_bits]
        else:
            cindices_list = []
    else:
        qindices_list = []
        cindices_list = []
    new_node = DAGDepNode(type='op', op=operation, name=operation.name, qargs=qargs, cargs=cargs, successors=[], predecessors=[], qindices=qindices_list, cindices=cindices_list)
    return new_node