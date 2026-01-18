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
def add_clbits(self, clbits):
    """Add individual clbit wires."""
    if any((not isinstance(clbit, Clbit) for clbit in clbits)):
        raise DAGDependencyError('not a Clbit instance.')
    duplicate_clbits = set(self.clbits).intersection(clbits)
    if duplicate_clbits:
        raise DAGDependencyError('duplicate clbits %s' % duplicate_clbits)
    self.clbits.extend(clbits)