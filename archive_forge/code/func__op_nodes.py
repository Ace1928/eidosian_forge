from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def _op_nodes(self):
    """Returns DAG nodes."""
    if not self.is_dag_dependency:
        return self.dag.op_nodes()
    else:
        return self.dag.get_nodes()