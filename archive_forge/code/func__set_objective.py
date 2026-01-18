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
def _set_objective(self, obj: ObjLinearExprT, minimize: bool):
    """Sets the objective of the model."""
    self.clear_objective()
    if isinstance(obj, IntVar):
        self.__model.objective.coeffs.append(1)
        self.__model.objective.offset = 0
        if minimize:
            self.__model.objective.vars.append(obj.index)
            self.__model.objective.scaling_factor = 1
        else:
            self.__model.objective.vars.append(self.negated(obj.index))
            self.__model.objective.scaling_factor = -1
    elif isinstance(obj, LinearExpr):
        coeffs_map, constant, is_integer = obj.get_float_var_value_map()
        if is_integer:
            if minimize:
                self.__model.objective.scaling_factor = 1
                self.__model.objective.offset = constant
            else:
                self.__model.objective.scaling_factor = -1
                self.__model.objective.offset = -constant
            for v, c in coeffs_map.items():
                self.__model.objective.coeffs.append(c)
                if minimize:
                    self.__model.objective.vars.append(v.index)
                else:
                    self.__model.objective.vars.append(self.negated(v.index))
        else:
            self.__model.floating_point_objective.maximize = not minimize
            self.__model.floating_point_objective.offset = constant
            for v, c in coeffs_map.items():
                self.__model.floating_point_objective.coeffs.append(c)
                self.__model.floating_point_objective.vars.append(v.index)
    elif isinstance(obj, numbers.Integral):
        self.__model.objective.offset = int(obj)
        self.__model.objective.scaling_factor = 1
    else:
        raise TypeError('TypeError: ' + str(obj) + ' is not a valid objective')