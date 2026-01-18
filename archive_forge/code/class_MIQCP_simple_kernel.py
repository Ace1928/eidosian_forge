import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary, maximize
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class MIQCP_simple_kernel(MIQCP_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=Binary)
        model.y = pmo.variable(domain=Binary)
        model.z = pmo.variable(domain=Binary)
        model.obj = pmo.objective(model.x, sense=maximize)
        model.c0 = pmo.constraint(model.x + model.y + model.z == 1)
        model.qc0 = pmo.constraint(model.x ** 2 + model.y ** 2 <= model.z ** 2)
        model.qc1 = pmo.constraint(model.x ** 2 <= model.y * model.z)