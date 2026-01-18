from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.basepasses import TransformationPass
def _check_inverse(self, node1, node2):
    """Checks whether op1 and op2 are inverse up to a phase, that is whether
        ``op2 = e^{i * d} op1^{-1})`` for some phase difference ``d``.
        If this is the case, we can replace ``op2 * op1`` by `e^{i * d} I``.
        The input to this function is a pair of DAG nodes.
        The output is a tuple representing whether the two nodes
        are inverse up to a phase and that phase difference.
        """
    phase_difference = 0
    if not self._matrix_based:
        is_inverse = node1.op.inverse() == node2.op
    elif len(node2.qargs) > self._max_qubits:
        is_inverse = False
    else:
        mat1 = Operator(node1.op.inverse()).data
        mat2 = Operator(node2.op).data
        props = {}
        is_inverse = matrix_equal(mat1, mat2, ignore_phase=True, props=props)
        if is_inverse:
            phase_difference = props['phase_difference']
    return (is_inverse, phase_difference)