from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches."""
    if len(op.wires) < EINSUM_OP_WIRECOUNT_PERF_THRESHOLD and math.ndim(state) < EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD or (op.batch_size and is_state_batched):
        return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
    return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)