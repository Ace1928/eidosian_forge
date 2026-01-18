from typing import Any, Union, Iterable
from fractions import Fraction
from decimal import Decimal
import numbers
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
def _approx_eq_iterables(val: Iterable, other: Iterable, *, atol: Union[int, float]) -> bool:
    """Iterates over arguments and calls approx_eq recursively.

    Types of `val` and `other` does not necessarily needs to match each other.
    They just need to be iterable of the same length and have the same
    structure, approx_eq() will be called on each consecutive element of `val`
    and `other`.

    Args:
        val: Source for approximate comparison.
        other: Target for approximate comparison.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details.

    Returns:
        True if objects are approximately equal, False otherwise. Returns
        NotImplemented when approximate equality is not implemented for given
        types.
    """
    iter1 = iter(val)
    iter2 = iter(other)
    done = object()
    cur_item1 = None
    while cur_item1 is not done:
        try:
            cur_item1 = next(iter1)
        except StopIteration:
            cur_item1 = done
        try:
            cur_item2 = next(iter2)
        except StopIteration:
            cur_item2 = done
        if not approx_eq(cur_item1, cur_item2, atol=atol):
            return False
    return True