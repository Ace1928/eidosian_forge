from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.suffix import import_suffix_generator
from pyomo.core.expr.numvalue import native_numeric_types, value
from pyomo.core.expr.visitor import evaluate_expression
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
import time
import logging
def _add_and_collect_column_data(self, var, obj_coef, constraints, coefficients):
    """
        Update the objective Pyomo objective function and constraints, and update
        the _vars_referenced_by Maps

        Returns the column and objective coefficient data to pass to the solver
        """
    if obj_coef.__class__ in native_numeric_types and obj_coef == 0.0:
        pass
    else:
        self._objective.expr += obj_coef * var
        self._vars_referenced_by_obj.add(var)
        obj_coef = _convert_to_const(obj_coef)
    coeff_list = list()
    constr_list = list()
    for val, c in zip(coefficients, constraints):
        c._body += val * var
        self._vars_referenced_by_con[c].add(var)
        cval = _convert_to_const(val)
        coeff_list.append(cval)
        constr_list.append(self._pyomo_con_to_solver_con_map[c])
    return (obj_coef, constr_list, coeff_list)