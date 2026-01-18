import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Reals
from pyomo.core.beta.list_objects import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData
class TestExpressionList(_TestComponentListBase, unittest.TestCase):
    _ctype = XExpressionList
    _cdatatype = _GeneralExpressionData

    def setUp(self):
        _TestComponentListBase.setUp(self)
        self._arg = lambda: self.model.x ** 3