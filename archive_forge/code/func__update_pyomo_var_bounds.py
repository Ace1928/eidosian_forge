from pyomo.contrib.appsi.base import PersistentBase
from pyomo.common.config import (
from .cmodel import cmodel, cmodel_available
from typing import List, Optional
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData, minimize, maximize
from pyomo.core.base.block import _BlockData
from pyomo.core.base import SymbolMap, TextLabeler
from pyomo.common.errors import InfeasibleConstraintException
def _update_pyomo_var_bounds(self):
    for cv, v in self._rvar_map.items():
        cv_lb = cv.get_lb()
        cv_ub = cv.get_ub()
        if -cmodel.inf < cv_lb:
            v.setlb(cv_lb)
            v_id = id(v)
            _v, _lb, _ub, _fixed, _domain, _value = self._vars[v_id]
            self._vars[v_id] = (_v, cv_lb, _ub, _fixed, _domain, _value)
        if cv_ub < cmodel.inf:
            v.setub(cv_ub)
            v_id = id(v)
            _v, _lb, _ub, _fixed, _domain, _value = self._vars[v_id]
            self._vars[v_id] = (_v, _lb, cv_ub, _fixed, _domain, _value)