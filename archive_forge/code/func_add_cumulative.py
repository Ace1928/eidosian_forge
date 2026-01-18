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
def add_cumulative(self, intervals: Iterable[IntervalVar], demands: Iterable[LinearExprT], capacity: LinearExprT) -> Constraint:
    """Adds Cumulative(intervals, demands, capacity).

        This constraint enforces that:

            for all t:
              sum(demands[i]
                if (start(intervals[i]) <= t < end(intervals[i])) and
                (intervals[i] is present)) <= capacity

        Args:
          intervals: The list of intervals.
          demands: The list of demands for each interval. Each demand must be >= 0.
            Each demand can be a 1-var affine expression (a * x + b).
          capacity: The maximum capacity of the cumulative constraint. It can be a
            1-var affine expression (a * x + b).

        Returns:
          An instance of the `Constraint` class.
        """
    cumulative = Constraint(self)
    model_ct = self.__model.constraints[cumulative.index]
    model_ct.cumulative.intervals.extend([self.get_interval_index(x) for x in intervals])
    for d in demands:
        model_ct.cumulative.demands.append(self.parse_linear_expression(d))
    model_ct.cumulative.capacity.CopyFrom(self.parse_linear_expression(capacity))
    return cumulative