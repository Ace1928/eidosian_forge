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
def activity(self, ct: LinearConstraint) -> np.double:
    """Returns the activity of a linear constraint after solve."""
    if not self.__solve_helper.has_solution():
        return pd.NA
    return self.__solve_helper.activity(ct.index)