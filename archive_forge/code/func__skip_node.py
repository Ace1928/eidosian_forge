from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.basepasses import TransformationPass
def _skip_node(self, node):
    """Returns True if we should skip this node for the analysis."""
    if not isinstance(node, DAGOpNode):
        return True
    if getattr(node.op, '_directive', False) or node.name in {'measure', 'reset', 'delay'}:
        return True
    if getattr(node.op, 'condition', None):
        return True
    if node.op.is_parameterized():
        return True
    return False