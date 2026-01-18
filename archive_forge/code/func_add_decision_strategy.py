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
def add_decision_strategy(self, variables: Sequence[IntVar], var_strategy: cp_model_pb2.DecisionStrategyProto.VariableSelectionStrategy, domain_strategy: cp_model_pb2.DecisionStrategyProto.DomainReductionStrategy) -> None:
    """Adds a search strategy to the model.

        Args:
          variables: a list of variables this strategy will assign.
          var_strategy: heuristic to choose the next variable to assign.
          domain_strategy: heuristic to reduce the domain of the selected variable.
            Currently, this is advanced code: the union of all strategies added to
            the model must be complete, i.e. instantiates all variables. Otherwise,
            solve() will fail.
        """
    strategy = self.__model.search_strategy.add()
    for v in variables:
        expr = strategy.exprs.add()
        if v.index >= 0:
            expr.vars.append(v.index)
            expr.coeffs.append(1)
        else:
            expr.vars.append(self.negated(v.index))
            expr.coeffs.append(-1)
            expr.offset = 1
    strategy.variable_selection_strategy = var_strategy
    strategy.domain_reduction_strategy = domain_strategy