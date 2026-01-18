from typing import Any, Callable, Iterable, Sequence, Tuple, Union, cast, List
from cirq import circuits, ops, protocols
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import (
def _sorted_best_string_placements(possible_nodes: Iterable[Any], output_ops: Sequence[ops.Operation], key: Callable[[Any], ops.PauliStringPhasor]=lambda node: node.val) -> List[Tuple[ops.PauliStringPhasor, int, circuitdag.Unique[ops.PauliStringPhasor]]]:
    sort_key = lambda placement: (-len(placement[0].pauli_string), placement[1])
    node_maxes = []
    for possible_node in possible_nodes:
        string_op = key(possible_node)
        node_max = (string_op, 0, possible_node)
        for i, out_op in enumerate(output_ops):
            if not set(out_op.qubits) & set(string_op.qubits):
                continue
            if isinstance(out_op, ops.PauliStringPhasor) and protocols.commutes(out_op.pauli_string, string_op.pauli_string):
                continue
            if not (isinstance(out_op, ops.GateOperation) and isinstance(out_op.gate, (ops.SingleQubitCliffordGate, ops.PauliInteractionGate, ops.CZPowGate))):
                break
            string_op = string_op.pass_operations_over([out_op], after_to_before=True)
            curr = (string_op, i + 1, possible_node)
            if sort_key(curr) > sort_key(node_max):
                node_max = curr
        node_maxes.append(node_max)
    return sorted(node_maxes, key=sort_key, reverse=True)