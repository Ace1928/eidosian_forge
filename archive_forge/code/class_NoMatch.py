from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
class NoMatch(ValueError):

    def __init__(self, data: str, index: int, expected: List[str]):
        self.data = data
        self.index = index
        self.expected = expected
        super(NoMatch, self).__init__(f'Can not match at index {index}. Got {data[index:index + 5]!r}, expected any of {expected}.\nContext(data[-10:+10]): {data[index - 10:index + 10]!r}')