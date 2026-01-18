from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from typing import TypeVar, Generic
from pyrsistent._plist import plist

        Return deque with elements rotated steps steps.

        >>> x = pdeque([1, 2, 3])
        >>> x.rotate(1)
        pdeque([3, 1, 2])
        >>> x.rotate(-2)
        pdeque([3, 1, 2])
        