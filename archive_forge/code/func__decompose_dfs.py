import itertools
import dataclasses
import inspect
from collections import defaultdict
from typing import (
from typing_extensions import runtime_checkable
from typing_extensions import Protocol
from cirq import devices, ops
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def _decompose_dfs(item: Any, args: _DecomposeArgs) -> Iterator['cirq.Operation']:
    from cirq.circuits import CircuitOperation, FrozenCircuit
    if isinstance(item, ops.Operation):
        item_untagged = item.untagged
        if args.preserve_structure and isinstance(item_untagged, CircuitOperation):
            new_fc = FrozenCircuit(_decompose_dfs(item_untagged.circuit, args))
            yield item_untagged.replace(circuit=new_fc).with_tags(*item.tags)
            return
        if args.keep is not None and args.keep(item):
            yield item
            return
    decomposed = _try_op_decomposer(item, args.intercepting_decomposer, context=args.context)
    if decomposed is NotImplemented or decomposed is None:
        decomposed = decompose_once(item, default=None, flatten=False, context=args.context)
    if decomposed is NotImplemented or decomposed is None:
        decomposed = _try_op_decomposer(item, args.fallback_decomposer, context=args.context)
    if decomposed is NotImplemented or decomposed is None:
        if not isinstance(item, ops.Operation) and isinstance(item, Iterable):
            decomposed = item
    if decomposed is NotImplemented or decomposed is None:
        if args.keep is not None and args.on_stuck_raise is not None:
            if isinstance(args.on_stuck_raise, Exception):
                raise args.on_stuck_raise
            elif callable(args.on_stuck_raise):
                error = args.on_stuck_raise(item)
                if error is not None:
                    raise error
        yield item
    else:
        for val in ops.flatten_to_ops(decomposed):
            yield from _decompose_dfs(val, args)