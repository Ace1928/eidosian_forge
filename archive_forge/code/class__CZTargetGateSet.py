from typing import Callable
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.pauli_string_optimize import pauli_string_optimized_circuit
from cirq.contrib.paulistring.clifford_optimize import clifford_optimized_circuit
class _CZTargetGateSet(transformers.CZTargetGateset):
    """Private implementation of `cirq.CZTargetGateset` used for optimized_circuit method below.

    The implementation extends `cirq.CZTargetGateset` by modifying decomposed operations using
    `post_clean_up` before putting them back in the circuit.
    """

    def __init__(self, post_clean_up: Callable[[ops.OP_TREE], ops.OP_TREE]=lambda op_tree: op_tree):
        super().__init__()
        self.post_clean_up = post_clean_up

    def _decompose_two_qubit_operation(self, op: ops.Operation, _) -> ops.OP_TREE:
        ret = super()._decompose_two_qubit_operation(op, _)
        return ret if ret is NotImplemented else self.post_clean_up(ret)