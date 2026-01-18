import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_block_kernel(LP_block):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.b = pmo.block()
        model.B = pmo.block_dict(((i, pmo.block()) for i in range(1, 4)))
        model.a = pmo.parameter(value=1.0)
        model.b.x = pmo.variable(lb=0)
        model.B[1].x = pmo.variable(lb=0)
        model.obj = pmo.objective(expr=model.b.x + 3.0 * model.B[1].x)
        model.obj.deactivate()
        model.B[2].c = pmo.constraint(expr=-model.B[1].x <= -model.a)
        model.B[2].obj = pmo.objective(expr=model.b.x + 3.0 * model.B[1].x + 2)
        model.B[3].c = pmo.constraint(expr=(2.0, model.b.x / model.a - model.B[1].x, 10))