import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
class _Sum(LinearExpr):
    """Represents the sum of two LinearExprs."""

    def __init__(self, left, right):
        for x in [left, right]:
            if not isinstance(x, (numbers.Number, LinearExpr)):
                raise TypeError('not an linear expression: ' + str(x))
        self.__left = left
        self.__right = right

    def left(self):
        return self.__left

    def right(self):
        return self.__right

    def __str__(self):
        return f'({self.__left} + {self.__right})'

    def __repr__(self):
        return f'sum({self.__left!r}, {self.__right!r})'