from collections.abc import Container, Iterable, Sized, Hashable
from functools import reduce
from typing import Generic, TypeVar
from pyrsistent._pmap import pmap

        Hash based on value of elements.

        >>> m = pmap({pbag([1, 2]): "it's here!"})
        >>> m[pbag([2, 1])]
        "it's here!"
        >>> pbag([1, 1, 2]) in m
        False
        