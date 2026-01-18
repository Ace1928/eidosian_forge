import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_duals_minimize_kernel(LP_duals_minimize):

    def _generate_model(self):
        self.model = None
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.s = list(range(1, 13))
        model.x = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
        model.x[1].lb = -1
        model.x[1].ub = 1
        model.x[2].lb = -1
        model.x[2].ub = 1
        model.obj = pmo.objective(sum((model.x[i] * (-1) ** (i + 1) for i in model.s)))
        model.c = pmo.constraint_dict()
        model.c[3] = pmo.constraint(model.x[3] >= -1.0)
        model.c[4] = pmo.constraint(model.x[4] <= 1.0)
        model.c[5] = pmo.constraint(model.x[5] == -1.0)
        model.c[6] = pmo.constraint(model.x[6] == -1.0)
        model.c[7] = pmo.constraint(model.x[7] == 1.0)
        model.c[8] = pmo.constraint(model.x[8] == 1.0)
        model.c[9] = pmo.constraint((-1.0, model.x[9], -1.0))
        model.c[10] = pmo.constraint((-1.0, model.x[10], -1.0))
        model.c[11] = pmo.constraint((1.0, model.x[11], 1.0))
        model.c[12] = pmo.constraint((1.0, model.x[12], 1.0))
        model.c_inactive = pmo.constraint_dict()
        model.c_inactive[3] = pmo.constraint(model.x[3] >= -2.0)
        model.c_inactive[4] = pmo.constraint(model.x[4] <= 2.0)