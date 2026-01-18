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
def _as_flat_linear_expression(base_expr: LinearExprT) -> _LinearExpression:
    """Converts floats, ints and Linear objects to a LinearExpression."""
    if isinstance(base_expr, _LinearExpression):
        return base_expr
    return _sum_as_flat_linear_expression(to_process=[(base_expr, 1.0)], offset=0.0)