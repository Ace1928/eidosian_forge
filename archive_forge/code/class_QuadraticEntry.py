import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
@dataclasses.dataclass(frozen=True)
class QuadraticEntry:
    """Represents an id-indexed quadratic term."""
    __slots__ = ('id_key', 'coefficient')
    id_key: QuadraticTermIdKey
    coefficient: float