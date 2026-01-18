from typing import Tuple
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def _pull_non_clifford_before(circuit: circuits.Circuit) -> ops.OP_TREE:

    def _iter_ops_range_reversed(moment_end):
        for i in reversed(range(moment_end)):
            moment = circuit[i]
            for op in moment.operations:
                if not isinstance(op, ops.PauliStringPhasor):
                    yield op
    for i, moment in enumerate(circuit):
        for op in moment.operations:
            if isinstance(op, ops.PauliStringPhasor):
                ops_to_cross = _iter_ops_range_reversed(i)
                yield op.pass_operations_over(ops_to_cross)