import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_constant_objective2_kernel(LP_constant_objective2):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=NonNegativeReals)
        model.obj = pmo.objective(model.x - model.x)
        model.con = pmo.constraint(model.x == 1.0)