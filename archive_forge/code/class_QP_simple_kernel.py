import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class QP_simple_kernel(QP_simple):

    def _generate_model(self):
        self.model = None
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.a = pmo.parameter(value=1.0)
        model.x = pmo.variable(domain=NonNegativeReals)
        model.y = pmo.variable(domain=NonNegativeReals)
        model.inactive_obj = pmo.objective(model.y)
        model.inactive_obj.deactivate()
        model.obj = pmo.objective(model.x ** 2 + 3.0 * model.inactive_obj ** 2 + 1.0)
        model.c1 = pmo.constraint(model.a <= model.y)
        model.c2 = pmo.constraint((2.0, model.x / model.a - model.y, 10))