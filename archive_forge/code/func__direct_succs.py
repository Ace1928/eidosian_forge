from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def _direct_succs(self, node):
    """Returns direct successors of a node. This function takes into account the
        direction of collecting blocks, that is node's successors when collecting
        backwards are the direct predecessors of a node in the DAG.
        """
    if not self.is_dag_dependency:
        if self._collect_from_back:
            return [succ for succ in self.dag.predecessors(node) if isinstance(succ, DAGOpNode)]
        else:
            return [succ for succ in self.dag.successors(node) if isinstance(succ, DAGOpNode)]
    elif self._collect_from_back:
        return [self.dag.get_node(succ_id) for succ_id in self.dag.direct_predecessors(node.node_id)]
    else:
        return [self.dag.get_node(succ_id) for succ_id in self.dag.direct_successors(node.node_id)]