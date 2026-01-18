import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_constant_objective1_kernel(LP_constant_objective1):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=NonNegativeReals)
        model.obj = pmo.objective(0.0)
        model.con = pmo.linear_constraint(terms=[(model.x, 1.0)], rhs=1.0)