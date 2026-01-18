import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_unbounded_kernel(LP_unbounded):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable()
        model.y = pmo.variable()
        model.o = pmo.objective(model.x + model.y)