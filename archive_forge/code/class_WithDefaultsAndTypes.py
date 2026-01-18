import asyncio
import functools
from typing import Tuple
class WithDefaultsAndTypes(object):
    """Class with functions that have default arguments and types."""

    def double(self, count: float=0) -> float:
        """Returns the input multiplied by 2.

    Args:
      count: Input number that you want to double.

    Returns:
      A number that is the double of count.
    """
        return 2 * count

    def get_int(self, value: int=None):
        return 0 if value is None else value