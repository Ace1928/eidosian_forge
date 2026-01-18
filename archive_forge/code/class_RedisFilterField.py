from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
class RedisFilterField:
    """Base class for RedisFilterFields."""
    escaper: 'TokenEscaper' = TokenEscaper()
    OPERATORS: Dict[RedisFilterOperator, str] = {}

    def __init__(self, field: str):
        self._field = field
        self._value: Any = None
        self._operator: RedisFilterOperator = RedisFilterOperator.EQ

    def equals(self, other: 'RedisFilterField') -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._field == other._field and self._value == other._value

    def _set_value(self, val: Any, val_type: Tuple[Any], operator: RedisFilterOperator) -> None:
        if operator not in self.OPERATORS:
            raise ValueError(f'Operator {operator} not supported by {self.__class__.__name__}. ' + f'Supported operators are {self.OPERATORS.values()}.')
        if not isinstance(val, val_type):
            raise TypeError(f'Right side argument passed to operator {self.OPERATORS[operator]} with left side argument {self.__class__.__name__} must be of type {val_type}, received value {val}')
        self._value = val
        self._operator = operator