import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _tuple_of_big_endian_int(bit_groups: Iterable[Any]) -> Tuple[int, ...]:
    """Returns the big-endian integers specified by groups of bits.

    Args:
        bit_groups: Groups of descending bits, each specifying a big endian
            integer with the 1s bit at the end.

    Returns:
        A tuple containing the integer for each group.
    """
    return tuple((value.big_endian_bits_to_int(bits) for bits in bit_groups))