from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _swap_symbols_to_sqrt_iswap(a: 'cirq.Qid', b: 'cirq.Qid', turns: 'cirq.TParamVal', use_sqrt_iswap_inv: bool=True):
    """Implements `cirq.SWAP(a, b) ** turns` using two √iSWAPs and single qubit rotations.

    Output unitary:
        [[1, 0,        0,     0],
         [0, g·c,    -i·g·s,  0],
         [0, -i·g·s,  g·c,    0],
         [0,   0,      0,     1]]
    where:
        c = cos(π·t/2), s = sin(π·t/2), g = exp(i·π·t/2).

    Args:
        a: The first qubit.
        b: The second qubit.
        turns: The rotational angle (t) that specifies the gate, where
            c = cos(π·t/2), s = sin(π·t/2), g = exp(i·π·t/2).
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    yield (ops.Z(a) ** 1.25)
    yield (ops.Z(b) ** (-0.25))
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield (ops.Z(a) ** (-turns / 2 + 1))
    yield (ops.Z(b) ** (turns / 2))
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield (ops.Z(a) ** (turns / 2 - 0.25))
    yield (ops.Z(b) ** (turns / 2 + 0.25))
    yield _cphase_symbols_to_sqrt_iswap(a, b, -turns, use_sqrt_iswap_inv)