import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
class SerializeMemoizer(Serialize):
    """A version of serialize that memoizes objects to reduce space"""
    __serialize_fields__ = ('memoized',)

    def __init__(self, types_to_memoize: List) -> None:
        self.types_to_memoize = tuple(types_to_memoize)
        self.memoized = Enumerator()

    def in_types(self, value: Serialize) -> bool:
        return isinstance(value, self.types_to_memoize)

    def serialize(self) -> Dict[int, Any]:
        return _serialize(self.memoized.reversed(), None)

    @classmethod
    def deserialize(cls, data: Dict[int, Any], namespace: Dict[str, Any], memo: Dict[Any, Any]) -> Dict[int, Any]:
        return _deserialize(data, namespace, memo)