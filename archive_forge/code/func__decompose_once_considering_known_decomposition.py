from typing import Any, FrozenSet, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def _decompose_once_considering_known_decomposition(val: Any) -> DecomposeResult:
    """Decomposes a value into operations, if possible.

    Args:
        val: The value to decompose into operations.

    Returns:
        A tuple of operations if decomposition succeeds.
    """
    import uuid
    context = cirq.DecompositionContext(qubit_manager=cirq.GreedyQubitManager(prefix=f'_{uuid.uuid4()}', maximize_reuse=True))
    decomposed = _try_decompose_from_known_decompositions(val, context)
    if decomposed is not None:
        return decomposed
    if isinstance(val, cirq.Gate):
        decomposed = cirq.decompose_once_with_qubits(val, cirq.LineQid.for_gate(val), context=context, flatten=False, default=None)
    else:
        decomposed = cirq.decompose_once(val, context=context, flatten=False, default=None)
    return [*cirq.flatten_to_ops(decomposed)] if decomposed is not None else None