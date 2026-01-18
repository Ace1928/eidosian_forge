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
class IntervalConfig(ConfigDict):
    """
    Attributes
    ----------
    feasibility_tol: float
    integer_tol: float
    improvement_tol: float
    max_iter: int
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super(IntervalConfig, self).__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.feasibility_tol: float = self.declare('feasibility_tol', ConfigValue(domain=NonNegativeFloat, default=1e-08))
        self.integer_tol: float = self.declare('integer_tol', ConfigValue(domain=NonNegativeFloat, default=1e-05))
        self.improvement_tol: float = self.declare('improvement_tol', ConfigValue(domain=NonNegativeFloat, default=0.0001))
        self.max_iter: int = self.declare('max_iter', ConfigValue(domain=NonNegativeInt, default=10))
        self.deactivate_satisfied_constraints: bool = self.declare('deactivate_satisfied_constraints', ConfigValue(domain=bool, default=False))