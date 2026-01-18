from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform

        Remove the first occurrence of a value from the vector.

        >>> v1 = v(1, 2, 3, 2, 1)
        >>> v2 = v1.remove(1)
        >>> v2
        pvector([2, 3, 2, 1])
        >>> v2.remove(1)
        pvector([2, 3, 2])
        