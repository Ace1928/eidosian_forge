import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
@dataclasses.dataclass(frozen=True)
class LinearObjectiveEntry:
    __slots__ = ('variable_id', 'coefficient')
    variable_id: int
    coefficient: float