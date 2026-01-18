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
def add_forbidden_assignments(self, variables: Sequence[VariableT], tuples_list: Iterable[Sequence[IntegralT]]) -> Constraint:
    """Adds add_forbidden_assignments(variables, [tuples_list]).

        A ForbiddenAssignments constraint is a constraint on an array of variables
        where the list of impossible combinations is provided in the tuples list.

        Args:
          variables: A list of variables.
          tuples_list: A list of forbidden tuples. Each tuple must have the same
            length as the variables, and the *i*th value of a tuple corresponds to
            the *i*th variable.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          TypeError: If a tuple does not have the same size as the list of
                     variables.
          ValueError: If the array of variables is empty.
        """
    if not variables:
        raise ValueError('add_forbidden_assignments expects a non-empty variables array')
    index = len(self.__model.constraints)
    ct = self.add_allowed_assignments(variables, tuples_list)
    self.__model.constraints[index].table.negated = True
    return ct