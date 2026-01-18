import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
class TestExpressionData(unittest.TestCase):

    def test_exprdata_get_set(self):
        model = ConcreteModel()
        model.e = Expression([1])
        self.assertEqual(len(model.e), 1)
        self.assertEqual(model.e[1].expr, None)
        model.e.add(1, 1)
        self.assertEqual(model.e[1].expr, 1)
        model.e[1].expr += 2
        self.assertEqual(model.e[1].expr, 3)

    def test_exprdata_get_set_value(self):
        model = ConcreteModel()
        model.e = Expression([1])
        self.assertEqual(len(model.e), 1)
        self.assertEqual(model.e[1].expr, None)
        model.e.add(1, 1)
        model.e[1].expr = 1
        self.assertEqual(model.e[1].expr, 1)
        model.e[1].expr += 2
        self.assertEqual(model.e[1].expr, 3)

    def test_copy(self):
        model = ConcreteModel()
        model.a = Var(initialize=5)
        model.b = Var(initialize=10)
        model.expr1 = Expression(initialize=model.a)
        expr2 = copy.copy(model.expr1)
        self.assertEqual(model.expr1(), 5)
        self.assertEqual(expr2(), 5)
        self.assertEqual(id(model.expr1.expr), id(expr2.expr))
        model.expr1.expr.set_value(1)
        self.assertEqual(model.expr1(), 1)
        self.assertEqual(expr2(), 1)
        self.assertEqual(id(model.expr1.expr), id(expr2.expr))
        model.expr1.set_value(model.b)
        self.assertEqual(model.expr1(), 10)
        self.assertEqual(expr2(), 1)
        self.assertNotEqual(id(model.expr1.expr), id(expr2.expr))
        model.a.set_value(5)
        model.b.set_value(10)
        model.del_component('expr1')
        model.expr1 = Expression(initialize=model.a + model.b)
        expr2 = copy.copy(model.expr1)
        self.assertEqual(model.expr1(), 15)
        self.assertEqual(expr2(), 15)
        self.assertEqual(id(model.expr1.expr), id(expr2.expr))
        self.assertEqual(id(model.expr1.expr.arg(0)), id(expr2.expr.arg(0)))
        self.assertEqual(id(model.expr1.expr.arg(1)), id(expr2.expr.arg(1)))
        model.a.set_value(0)
        self.assertEqual(model.expr1(), 10)
        self.assertEqual(expr2(), 10)
        self.assertEqual(id(model.expr1.expr), id(expr2.expr))
        self.assertEqual(id(model.expr1.expr.arg(0)), id(expr2.expr.arg(0)))
        self.assertEqual(id(model.expr1.expr.arg(1)), id(expr2.expr.arg(1)))
        model.expr1.expr += 1
        self.assertEqual(model.expr1(), 11)
        self.assertEqual(expr2(), 10)
        self.assertNotEqual(id(model.expr1.expr), id(expr2.expr))

    def test_model_clone(self):
        model = ConcreteModel()
        model.x = Var(initialize=2.0)
        model.y = Var(initialize=0.0)
        model.ec = Expression(initialize=model.x ** 2 + 1)
        model.obj = Objective(expr=model.y + model.ec)
        self.assertEqual(model.obj.expr(), 5.0)
        self.assertTrue(id(model.ec) in [id(e) for e in model.obj.expr.args])
        inst = model.clone()
        self.assertEqual(inst.obj.expr(), 5.0)
        if not id(inst.ec) in [id(e) for e in inst.obj.expr.args]:
            print('BUG?')
            print(id(inst.ec))
            print(inst.obj.expr.__class__)
            print([id(e) for e in inst.obj.expr.args])
            print([e.__class__ for e in inst.obj.expr.args])
            print([id(e) for e in model.obj.expr.args])
            print([e.__class__ for e in model.obj.expr.args])
        self.assertTrue(id(inst.ec) in [id(e) for e in inst.obj.expr.args])
        self.assertNotEqual(id(model.ec), id(inst.ec))
        self.assertFalse(id(inst.ec) in [id(e) for e in model.obj.expr.args])

    def test_is_constant(self):
        model = ConcreteModel()
        model.x = Var(initialize=1.0)
        model.p = Param(initialize=1.0)
        model.ec = Expression(initialize=model.x)
        self.assertEqual(model.ec.is_constant(), False)
        self.assertEqual(model.ec.expr.is_constant(), False)
        model.ec.set_value(model.p)
        self.assertEqual(model.ec.is_constant(), False)
        self.assertEqual(model.ec.expr.is_constant(), True)

    def test_polynomial_degree(self):
        model = ConcreteModel()
        model.x = Var(initialize=1.0)
        model.ec = Expression(initialize=model.x)
        self.assertEqual(model.ec.polynomial_degree(), model.ec.expr.polynomial_degree())
        self.assertEqual(model.ec.polynomial_degree(), 1)
        model.ec.set_value(model.x ** 2)
        self.assertEqual(model.ec.polynomial_degree(), model.ec.expr.polynomial_degree())
        self.assertEqual(model.ec.polynomial_degree(), 2)

    def test_init_concrete(self):
        model = ConcreteModel()
        model.y = Var(initialize=0.0)
        model.x = Var(initialize=1.0)
        model.ec = Expression(expr=0)
        model.obj = Objective(expr=1.0 + model.ec)
        self.assertEqual(model.obj.expr(), 1.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(), 2.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(), 3.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(), 3.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=model.y)
        model.obj = Objective(expr=1.0 + model.ec)
        self.assertEqual(model.obj.expr(), 1.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(), 2.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(), 3.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(), 3.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        model.del_component('obj')
        model.del_component('ec')
        model.y.set_value(-1)
        model.ec = Expression(initialize=model.y + 1.0)
        model.obj = Objective(expr=1.0 + model.ec)
        self.assertEqual(model.obj.expr(), 1.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(), 2.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(), 3.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(), 3.0)
        self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))

    def test_init_abstract(self):
        model = AbstractModel()
        model.y = Var(initialize=0.0)
        model.x = Var(initialize=1.0)
        model.ec = Expression(initialize=0.0)

        def obj_rule(model):
            return 1.0 + model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(), 1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(), 2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(), 3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(), 3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=0.0)

        def obj_rule(model):
            return 1.0 + model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(), 1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(), 2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(), 3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(), 3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=0.0)

        def obj_rule(model):
            return 1.0 + model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(), 1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(), 2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(), 3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(), 3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))