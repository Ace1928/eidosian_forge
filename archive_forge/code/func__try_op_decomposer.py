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
def _try_op_decomposer(val: Any, decomposer: Optional[OpDecomposer], *, context: Optional[DecompositionContext]=None) -> DecomposeResult:
    if decomposer is None or not isinstance(val, ops.Operation):
        return None
    if 'context' in inspect.signature(decomposer).parameters:
        assert isinstance(decomposer, OpDecomposerWithContext)
        return decomposer(val, context=context)
    else:
        return decomposer(val)