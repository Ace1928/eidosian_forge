import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_unused_vars_kernel(LP_unused_vars):

    def _generate_model(self):
        self.model = None
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.s = [1, 2]
        model.x_unused = pmo.variable()
        model.x_unused.stale = False
        model.x_unused_initially_stale = pmo.variable()
        model.x_unused_initially_stale.stale = True
        model.X_unused = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
        model.X_unused_initially_stale = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
        for i in model.X_unused:
            model.X_unused[i].stale = False
            model.X_unused_initially_stale[i].stale = True
        model.x = pmo.variable()
        model.x.stale = False
        model.x_initially_stale = pmo.variable()
        model.x_initially_stale.stale = True
        model.X = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
        model.X_initially_stale = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
        for i in model.X:
            model.X[i].stale = False
            model.X_initially_stale[i].stale = True
        model.obj = pmo.objective(model.x + model.x_initially_stale + sum(model.X.values()) + sum(model.X_initially_stale.values()))
        model.c = pmo.constraint_dict()
        model.c[1] = pmo.constraint(model.x >= 1)
        model.c[2] = pmo.constraint(model.x_initially_stale >= 1)
        model.c[3] = pmo.constraint(model.X[1] >= 0)
        model.c[4] = pmo.constraint(model.X[2] >= 1)
        model.c[5] = pmo.constraint(model.X_initially_stale[1] >= 0)
        model.c[6] = pmo.constraint(model.X_initially_stale[2] >= 1)
        flat_model = model.clone()
        model.b = pmo.block()
        model.B = pmo.block_dict()
        model.B[1] = pmo.block()
        model.B[2] = pmo.block()
        model.b.b = flat_model.clone()
        model.B[1].b = flat_model.clone()
        model.B[2].b = flat_model.clone()
        model.b.deactivate()
        model.B.deactivate(shallow=False)
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False