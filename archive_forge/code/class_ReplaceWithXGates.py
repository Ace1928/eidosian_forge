from typing import Optional, TYPE_CHECKING, Set, List
import pytest
import cirq
from cirq import PointOptimizer, PointOptimizationSummary, Operation
from cirq.testing import EqualsTester
class ReplaceWithXGates(PointOptimizer):
    """Replaces a block of operations with X gates.

    Searches ahead for gates covering a subset of the focused operation's
    qubits, clears the whole range, and inserts X gates for each cleared
    operation's qubits.
    """

    def optimization_at(self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation') -> Optional['cirq.PointOptimizationSummary']:
        end = index + 1
        new_ops = [cirq.X(q) for q in op.qubits]
        done = False
        while not done:
            n = circuit.next_moment_operating_on(op.qubits, end)
            if n is None:
                break
            next_ops: Set[Optional[Operation]] = {circuit.operation_at(q, n) for q in op.qubits}
            next_ops_list: List[Operation] = [e for e in next_ops if e]
            next_ops_sorted = sorted(next_ops_list, key=lambda e: str(e.qubits))
            for next_op in next_ops_sorted:
                if next_op:
                    if set(next_op.qubits).issubset(op.qubits):
                        end = n + 1
                        new_ops.extend((cirq.X(q) for q in next_op.qubits))
                    else:
                        done = True
        return PointOptimizationSummary(clear_span=end - index, clear_qubits=op.qubits, new_operations=new_ops)