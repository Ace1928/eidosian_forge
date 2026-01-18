import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_trivial_constraints_kernel(LP_trivial_constraints):

    def _generate_model(self):
        self.model = None
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=RangeSet(float('-inf'), None, 0))
        model.y = pmo.variable(ub=float('inf'))
        model.obj = pmo.objective(model.x - model.y)
        model.c = pmo.constraint_dict()
        model.c[1] = pmo.constraint(model.x >= -2)
        model.c[2] = pmo.constraint(model.y <= 3)
        cdata = model.c[3] = pmo.constraint((0, 1, 3))
        assert cdata.lb == 0
        assert cdata.ub == 3
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c[4] = pmo.constraint((0, 2, 3))
        assert cdata.lb == 0
        assert cdata.ub == 3
        assert cdata.body() == 2
        assert not cdata.equality
        cdata = model.c[5] = pmo.constraint((0, 1, None))
        assert cdata.lb is None
        assert cdata.ub == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c[6] = pmo.constraint((None, 0, 1))
        assert cdata.lb is None
        assert cdata.ub == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c[7] = pmo.constraint((1, 1))
        assert cdata.lb == 1
        assert cdata.ub == 1
        assert cdata.body() == 1
        assert cdata.equality

    def post_solve_test_validation(self, tester, results):
        symbol_map = results.Solution(0).symbol_map
        assert not symbol_map is None
        if tester is None:
            for i in self.model.c:
                assert id(self.model.c[i]) in symbol_map.byObject
        else:
            tester.assertNotEqual(symbol_map, None)
            for i in self.model.c:
                tester.assertTrue(id(self.model.c[i]) in symbol_map.byObject)