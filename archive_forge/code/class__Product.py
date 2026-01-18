import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class _Product(LinearExpr):
    """Represents the (deferred) product of an expression by a constant."""
    __slots__ = ('_expression', '_coefficient')
    _expression: LinearExpr
    _coefficient: NumberT

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(_as_flat_linear_expression(self))