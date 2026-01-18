from collections.abc import Sequence, Hashable
from numbers import Integral
from functools import reduce
from typing import Generic, TypeVar

        Return new list with first element equal to elem removed. O(k) where k is the position
        of the element that is removed.

        Raises ValueError if no matching element is found.

        >>> plist([1, 2, 1]).remove(1)
        plist([2, 1])
        