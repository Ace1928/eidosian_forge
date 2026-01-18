from typing import (
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import common_gates, raw_types, global_phase_op
def _gen_gray_code(n: int) -> Iterator[Tuple[int, int]]:
    """Generate the Gray Code from 0 to 2^n-1.

    Each iteration yields a two-tuple, `(gray_code, bit_flip)`. `gray_code` is the decimal
    representation of the gray code and `bit_flip` is the position of bits flipped for next
    gray code.
    """
    gray_code = 0
    for i in range(1, 2 ** n):
        next_gray = i ^ i >> 1
        bit_flip = int(np.log2(gray_code ^ next_gray))
        yield (gray_code, bit_flip)
        gray_code = next_gray
    yield (gray_code, int(np.log2(gray_code)))