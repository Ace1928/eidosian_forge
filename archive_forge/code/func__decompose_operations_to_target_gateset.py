from typing import Optional, Callable, Hashable, Sequence, TYPE_CHECKING
from cirq import circuits
from cirq.protocols import decompose_protocol as dp
from cirq.transformers import transformer_api, transformer_primitives
@transformer_api.transformer
def _decompose_operations_to_target_gateset(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, gateset: Optional['cirq.Gateset']=None, decomposer: Callable[['cirq.Operation', int], dp.DecomposeResult]=lambda *_: NotImplemented, ignore_failures: bool=True, tags_to_decompose: Sequence[Hashable]=()) -> 'cirq.Circuit':
    """Decomposes every operation to `gateset` using `cirq.decompose` and `decomposer`.

    This transformer attempts to decompose every operation `op` in the given circuit to `gateset`
    using `cirq.decompose` protocol with `decomposer` used as an intercepting decomposer. This
    ensures that `op` is recursively decomposed using implicitly defined known decompositions
    (eg: in `_decompose_` magic method on the gaet class) till either `decomposer` knows how to
    decompose the given operation or the given operation belongs to `gateset`.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        gateset: Target gateset, which the decomposed operations should belong to.
        decomposer: A callable type which accepts an (operation, moment_index) and returns
            - An equivalent `cirq.OP_TREE` implementing `op` using gates from `gateset`.
            - `None` or `NotImplemented` if does not know how to decompose a given `op`.
        ignore_failures: If set, operations that fail to convert are left unchanged. If not set,
            conversion failures raise a ValueError.
        tags_to_decompose: `cirq.CircuitOperation`s tagged with any of `tags_to_decompose` will
            be decomposed even if context.deep is True.

    Returns:
        An equivalent circuit containing gates accepted by `gateset`.

    Raises:
        ValueError: If any input operation fails to convert and `ignore_failures` is False.
    """

    def map_func(op: 'cirq.Operation', moment_index: int):
        if context and context.deep and isinstance(op.untagged, circuits.CircuitOperation) and set(op.tags).isdisjoint(tags_to_decompose):
            return op
        return dp.decompose(op, intercepting_decomposer=lambda o: decomposer(o, moment_index), keep=gateset.validate if gateset else None, on_stuck_raise=None if ignore_failures or gateset is None else _create_on_stuck_raise_error(gateset))
    return transformer_primitives.map_operations_and_unroll(circuit, map_func, tags_to_ignore=context.tags_to_ignore if context else (), deep=context.deep if context else False).unfreeze(copy=False)