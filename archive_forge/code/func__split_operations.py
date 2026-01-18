from typing import Sequence, Callable
import pennylane as qml
from .core import transform
def _split_operations(ops, num_tapes):
    """
    Given a list of operators, return a list containing lists
    of new operators with length num_tapes, with the parameters split.
    """
    new_ops = [[] for _ in range(num_tapes)]
    for op in ops:
        if op.batch_size:
            for b in range(num_tapes):
                new_params = tuple((p if qml.math.ndim(p) == op.ndim_params[j] else p[b] for j, p in enumerate(op.data)))
                new_op = qml.ops.functions.bind_new_parameters(op, new_params)
                new_ops[b].append(new_op)
        else:
            for b in range(num_tapes):
                new_ops[b].append(op)
    return new_ops