from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, Union, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike, NDArray
def _flatten_to_ints(arg: ShapeInput) -> Iterable[int]:
    """
    Yield one integer at a time.

    Args:
        arg: Integers or iterables of integers, possibly nested, to be yielded.

    Yields:
        The provided integers in depth-first recursive order.

    Raises:
        ValueError: If an input is not an iterable or an integer.
    """
    for item in arg:
        try:
            if isinstance(item, Iterable):
                yield from _flatten_to_ints(item)
            elif int(item) == item:
                yield int(item)
            else:
                raise ValueError(f'Expected {item} to be iterable or an integer.')
        except (TypeError, RecursionError) as ex:
            raise ValueError(f'Expected {item} to be iterable or an integer.') from ex