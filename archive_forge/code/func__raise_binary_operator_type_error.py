import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def _raise_binary_operator_type_error(operator: str, lhs: Type[Any], rhs: Type[Any], extra_message: Optional[str]=None) -> NoReturn:
    """Raises TypeError on unsupported operators."""
    message = f'unsupported operand type(s) for {operator}: {lhs.__name__!r} and {rhs.__name__!r}'
    if extra_message is not None:
        message += '\n' + extra_message
    raise TypeError(message)