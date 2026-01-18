import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
def _gray_code_comparator(k1: Tuple[int, ...], k2: Tuple[int, ...], flip: bool=False) -> int:
    """Compares two Gray-encoded binary numbers.

    Args:
        k1: A tuple of ints, representing the bits that are one. For example, 6 would be (1, 2).
        k2: The second number, represented similarly as k1.
        flip: Whether to flip the comparison.

    Returns:
        -1 if k1 < k2 (or +1 if flip is true)
        0 if k1 == k2
        +1 if k1 > k2 (or -1 if flip is true)
    """
    max_1 = k1[-1] if k1 else -1
    max_2 = k2[-1] if k2 else -1
    if max_1 != max_2:
        return -1 if (max_1 < max_2) ^ flip else 1
    if max_1 == -1:
        return 0
    return _gray_code_comparator(k1[0:-1], k2[0:-1], not flip)