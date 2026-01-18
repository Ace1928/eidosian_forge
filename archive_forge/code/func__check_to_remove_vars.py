import abc
from typing import List
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.contrib.solver.util import collect_vars_and_named_exprs, get_objective
def _check_to_remove_vars(self, variables: List[_GeneralVarData]):
    vars_to_remove = {}
    for v in variables:
        v_id = id(v)
        ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
        if len(ref_cons) == 0 and len(ref_sos) == 0 and (ref_obj is None):
            vars_to_remove[v_id] = v
    self.remove_variables(list(vars_to_remove.values()))