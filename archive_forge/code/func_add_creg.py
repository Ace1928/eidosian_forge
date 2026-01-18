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
def add_creg(self, creg):
    """Add clbits in a classical register."""
    if not isinstance(creg, ClassicalRegister):
        raise DAGDependencyError('not a ClassicalRegister instance.')
    if creg.name in self.cregs:
        raise DAGDependencyError('duplicate register %s' % creg.name)
    self.cregs[creg.name] = creg
    existing_clbits = set(self.clbits)
    for j in range(creg.size):
        if creg[j] not in existing_clbits:
            self.clbits.append(creg[j])