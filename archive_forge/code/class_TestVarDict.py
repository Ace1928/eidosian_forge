import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Reals
from pyomo.core.beta.dict_objects import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData
class TestVarDict(_TestComponentDictBase, unittest.TestCase):
    _ctype = VarDict
    _cdatatype = lambda self, arg: _GeneralVarData()

    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self._arg = lambda: Reals