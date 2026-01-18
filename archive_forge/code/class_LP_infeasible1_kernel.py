import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_infeasible1_kernel(LP_infeasible1):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(lb=1)
        model.y = pmo.variable(lb=1)
        model.o = pmo.objective(model.x + model.y)
        model.c = pmo.constraint(model.x + model.y <= 0)