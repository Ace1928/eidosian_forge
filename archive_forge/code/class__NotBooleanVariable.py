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
class _NotBooleanVariable(LinearExpr):
    """Negation of a boolean variable."""

    def __init__(self, boolvar: IntVar):
        self.__boolvar: IntVar = boolvar

    @property
    def index(self) -> int:
        return -self.__boolvar.index - 1

    def negated(self) -> IntVar:
        return self.__boolvar

    def __invert__(self) -> IntVar:
        """Returns the logical negation of a Boolean literal."""
        return self.negated()

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return 'not(%s)' % str(self.__boolvar)

    def __bool__(self) -> NoReturn:
        raise NotImplementedError('Evaluating a literal as a Boolean value is not implemented.')

    def Not(self) -> 'IntVar':
        return self.negated()

    def Index(self) -> int:
        return self.index