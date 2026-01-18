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
def add_division_equality(self, target: LinearExprT, num: LinearExprT, denom: LinearExprT) -> Constraint:
    """Adds `target == num // denom` (integer division rounded towards 0)."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.int_div.exprs.append(self.parse_linear_expression(num))
    model_ct.int_div.exprs.append(self.parse_linear_expression(denom))
    model_ct.int_div.target.CopyFrom(self.parse_linear_expression(target))
    return ct