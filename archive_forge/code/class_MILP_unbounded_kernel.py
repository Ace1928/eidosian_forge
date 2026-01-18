import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Integers
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class MILP_unbounded_kernel(MILP_unbounded):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=pmo.IntegerSet)
        model.y = pmo.variable(domain=pmo.IntegerSet)
        model.o = pmo.objective(model.x + model.y)