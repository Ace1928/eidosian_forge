import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Reals
from pyomo.core.beta.list_objects import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData
class TestConstraintList(_TestActiveComponentListBase, unittest.TestCase):
    _ctype = XConstraintList
    _cdatatype = _GeneralConstraintData

    def setUp(self):
        _TestComponentListBase.setUp(self)
        self._arg = lambda: self.model.x >= 1