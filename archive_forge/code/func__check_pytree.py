from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_pytree(op):
    """Check that the operator is a pytree."""
    data, metadata = op._flatten()
    try:
        assert hash(metadata), 'metadata must be hashable'
    except Exception as e:
        raise AssertionError(f'metadata output from _flatten must be hashable. Got metadata {metadata}') from e
    try:
        new_op = type(op)._unflatten(data, metadata)
    except Exception as e:
        message = f'{type(op).__name__}._unflatten must be able to reproduce the original operation from {data} and {metadata}. You may need to override either the _unflatten or _flatten method. \nFor local testing, try type(op)._unflatten(*op._flatten())'
        raise AssertionError(message) from e
    assert op == new_op, 'metadata and data must be able to reproduce the original operation'
    try:
        import jax
    except ImportError:
        return
    leaves, struct = jax.tree_util.tree_flatten(op)
    unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)
    assert unflattened_op == op, f'op must be a valid pytree. Got {unflattened_op} instead of {op}.'
    for d1, d2 in zip(op.data, leaves):
        assert qml.math.allclose(d1, d2), f'data must be the terminal leaves of the pytree. Got {d1}, {d2}'