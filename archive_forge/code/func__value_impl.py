import abc
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import interface as intf
from cvxpy import utilities as u
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities.deterministic import unique_list
def _value_impl(self):
    if 0 in self.shape:
        result = np.array([])
    else:
        arg_values = []
        for arg in self.args:
            arg_val = arg._value_impl()
            if arg_val is None and (not self.is_constant()):
                return None
            else:
                arg_values.append(arg_val)
        result = self.numeric(arg_values)
    return result