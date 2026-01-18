import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class WalkerTests2(unittest.TestCase):

    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()
        e = sin(M.x) + M.x * M.y + 3
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * (2 * M.w[2]) + 3, f)

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        e = M.x
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, M.x, e)
        assertExpressionsEqual(self, 2 * M.w[1], f)

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()
        e = sin(M.x) + M.x * M.y + 3 <= 0
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * (2 * M.w[2]) + 3 <= 0, f)

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()
        e = inequality(0, sin(M.x) + M.x * M.y + 3, 1)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, inequality(0, sin(M.x) + M.x * M.y + 3, 1), e)
        assertExpressionsEqual(self, inequality(0, sin(2 * M.w[1]) + 2 * M.w[1] * (2 * M.w[2]) + 3, 1), f)

    def test_replacement_walker5(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)
        e = M.z * M.x
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, e, MonomialTermExpression((M.z, M.x)))
        assertExpressionsEqual(self, f, MonomialTermExpression((NPV_ProductExpression((M.z, 2)), M.w[1])))

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)
        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, e, LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]))
        assertExpressionsEqual(self, f, LinearExpression([MonomialTermExpression((NPV_ProductExpression((M.z[0], 2)), M.w[1])), MonomialTermExpression((NPV_ProductExpression((M.z[1], 2)), M.w[2])), MonomialTermExpression((NPV_ProductExpression((M.z[2], 2)), M.w[3]))]))
        e = 2 * sum_product(M.z, M.x)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, 2 * LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
        assertExpressionsEqual(self, f, ProductExpression((2, LinearExpression([MonomialTermExpression((NPV_ProductExpression((M.z[0], 2)), M.w[4])), MonomialTermExpression((NPV_ProductExpression((M.z[1], 2)), M.w[5])), MonomialTermExpression((NPV_ProductExpression((M.z[2], 2)), M.w[6]))]))))