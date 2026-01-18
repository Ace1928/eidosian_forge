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
def _deactivate_satisfied_cons(self):
    cons_to_deactivate = list()
    if self.config.deactivate_satisfied_constraints:
        for c, cc in self._con_map.items():
            if not cc.active:
                cons_to_deactivate.append(c)
    self.remove_constraints(cons_to_deactivate)
    for c in cons_to_deactivate:
        c.deactivate()