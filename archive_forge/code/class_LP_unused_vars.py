import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_unused_vars(_BaseTestModel):
    """
    A continuous linear model where some vars aren't used
    and some used vars start out with the stale flag as True
    """
    description = 'LP_unused_vars'
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.disable_suffix_tests = True
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.s = Set(initialize=[1, 2])
        model.x_unused = Var()
        model.x_unused.stale = False
        model.x_unused_initially_stale = Var()
        model.x_unused_initially_stale.stale = True
        model.X_unused = Var(model.s)
        model.X_unused_initially_stale = Var(model.s)
        for i in model.s:
            model.X_unused[i].stale = False
            model.X_unused_initially_stale[i].stale = True
        model.x = Var()
        model.x.stale = False
        model.x_initially_stale = Var()
        model.x_initially_stale.stale = True
        model.X = Var(model.s)
        model.X_initially_stale = Var(model.s)
        for i in model.s:
            model.X[i].stale = False
            model.X_initially_stale[i].stale = True
        model.obj = Objective(expr=model.x + model.x_initially_stale + sum_product(model.X) + sum_product(model.X_initially_stale))
        model.c = ConstraintList()
        model.c.add(model.x >= 1)
        model.c.add(model.x_initially_stale >= 1)
        model.c.add(model.X[1] >= 0)
        model.c.add(model.X[2] >= 1)
        model.c.add(model.X_initially_stale[1] >= 0)
        model.c.add(model.X_initially_stale[2] >= 1)
        flat_model = model.clone()
        model.b = Block()
        model.B = Block(model.s)
        model.b.b = flat_model.clone()
        model.B[1].b = flat_model.clone()
        model.B[2].b = flat_model.clone()
        model.b.deactivate()
        model.B.deactivate()
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x_unused.value = -1.0
        model.x_unused_initially_stale.value = -1.0
        model.x_unused_initially_stale.stale = True
        for i in model.s:
            model.X_unused[i].value = -1.0
            model.X_unused_initially_stale[i].value = -1.0
            model.X_unused_initially_stale[i].stale = True
        model.x.value = -1.0
        model.x_initially_stale.value = -1.0
        model.x_initially_stale.stale = True
        for i in model.s:
            model.X[i].value = -1.0
            model.X_initially_stale[i].value = -1.0
            model.X_initially_stale[i].stale = True