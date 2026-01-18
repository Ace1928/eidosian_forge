import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
def _index_is_valid(self, idx, allow_int_only=False):
    """
        Object to be used as list index is within range of
        list contained within self.

        Parameters
        ----------
        idx : object
            List index. Usually an integer type or slice.
        allow_int_only : bool, optional
            Being an integral type is a necessary condition
            for validity. The default is True.

        Returns
        -------
        : bool
            True if index is valid, False otherwise.
        """
    try:
        self._list[idx]
    except (TypeError, IndexError):
        slice_valid = False
    else:
        slice_valid = True
    int_req_satisfied = not allow_int_only or isinstance(idx, Integral)
    return slice_valid and int_req_satisfied