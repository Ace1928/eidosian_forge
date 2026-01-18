import asyncio
import functools
from typing import Tuple
class WithTypes(object):
    """Class with functions that have default arguments and types."""

    def double(self, count: float) -> float:
        """Returns the input multiplied by 2.

    Args:
      count: Input number that you want to double.

    Returns:
      A number that is the double of count.
    """
        return 2 * count

    def long_type(self, long_obj: Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[int]]]]]]]]]]]]):
        return long_obj