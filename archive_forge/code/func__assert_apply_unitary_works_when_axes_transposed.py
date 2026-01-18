from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def _assert_apply_unitary_works_when_axes_transposed(val: Any, *, atol: float=1e-08) -> None:
    """Tests whether a value's _apply_unitary_ handles out-of-order axes.

    A common mistake to make when implementing `_apply_unitary_` is to assume
    that the incoming axes will be contiguous, or ascending, or that they can be
    flattened, or that other axes have a length of two, etc, etc ,etc. This
    method checks that `_apply_unitary_` does the same thing to out-of-order
    axes that it does to contiguous in-order axes.

    Args:
        val: The operation, gate, or other unitary object to test.
        atol: Absolute error tolerance.

    Raises:
        AssertionError: If `_apply_unitary_` acted differently on the
            out-of-order axes than on the in-order axes.
    """
    if not hasattr(val, '_apply_unitary_') or not protocols.has_unitary(val):
        return
    shape = protocols.qid_shape(val)
    n = len(shape)
    padded_shape = shape + (1, 2, 2, 3)
    padded_n = len(padded_shape)
    size = np.prod(padded_shape, dtype=np.int64).item()
    permutation = list(range(padded_n))
    random.shuffle(permutation)
    transposed_shape = [0] * padded_n
    for i in range(padded_n):
        transposed_shape[permutation[i]] = padded_shape[i]
    in_order_input = lin_alg_utils.random_superposition(size).reshape(padded_shape)
    out_of_order_input = np.empty(shape=transposed_shape, dtype=np.complex128)
    out_of_order_input.transpose(permutation)[...] = in_order_input
    in_order_output = protocols.apply_unitary(val, protocols.ApplyUnitaryArgs(target_tensor=in_order_input, available_buffer=np.empty_like(in_order_input), axes=range(n)))
    out_of_order_output = protocols.apply_unitary(val, protocols.ApplyUnitaryArgs(target_tensor=out_of_order_input, available_buffer=np.empty_like(out_of_order_input), axes=permutation[:n]))
    reordered_output = out_of_order_output.transpose(permutation)
    if not np.allclose(in_order_output, reordered_output, atol=atol):
        raise AssertionError(f'The _apply_unitary_ method of {repr(val)} acted differently on out-of-order axes than on in-order axes.\n\nThe failing axis order: {repr(permutation[:n])}')