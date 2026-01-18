from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def _have_uncollected_nodes(self):
    """Returns whether there are uncollected (pending) nodes"""
    return len(self._pending_nodes) > 0