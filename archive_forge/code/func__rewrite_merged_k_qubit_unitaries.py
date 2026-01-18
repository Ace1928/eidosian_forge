from typing import cast, Optional, Callable, TYPE_CHECKING
from cirq import ops, protocols, circuits
from cirq.transformers import transformer_api, transformer_primitives
def _rewrite_merged_k_qubit_unitaries(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, k: int=0, rewriter: Optional[Callable[['cirq.CircuitOperation'], 'cirq.OP_TREE']]=None, merged_circuit_op_tag: str='_merged_k_qubit_unitaries_component') -> 'cirq.Circuit':
    deep = context.deep if context else False

    def map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        op_untagged = op.untagged
        if deep and isinstance(op_untagged, circuits.CircuitOperation) and (merged_circuit_op_tag not in op.tags):
            return op_untagged.replace(circuit=_rewrite_merged_k_qubit_unitaries(op_untagged.circuit, context=context, k=k, rewriter=rewriter, merged_circuit_op_tag=merged_circuit_op_tag).freeze()).with_tags(*op.tags)
        if not (protocols.num_qubits(op) <= k and protocols.has_unitary(op)):
            return op
        if rewriter:
            return rewriter(cast(circuits.CircuitOperation, op_untagged) if merged_circuit_op_tag in op.tags else circuits.CircuitOperation(circuits.FrozenCircuit(op)))
        return ops.MatrixGate(protocols.unitary(op)).on(*op.qubits)
    return transformer_primitives.map_operations_and_unroll(circuit, map_func, tags_to_ignore=context.tags_to_ignore if context else ()).unfreeze(copy=False)