import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary, Integers
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class MILP_discrete_var_bounds_kernel(MILP_discrete_var_bounds):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.w2 = pmo.variable(domain=pmo.BooleanSet)
        model.x2 = pmo.variable(domain_type=pmo.IntegerSet, lb=0, ub=1)
        model.yb = pmo.variable(domain_type=pmo.IntegerSet, lb=1, ub=1)
        model.zb = pmo.variable(domain_type=pmo.IntegerSet, lb=0, ub=0)
        model.yi = pmo.variable(domain=pmo.IntegerSet, lb=-1)
        model.zi = pmo.variable(domain=pmo.IntegerSet, ub=1)
        model.obj = pmo.objective(model.w2 - model.x2 + model.yb - model.zb + model.yi - model.zi)
        model.c3 = pmo.constraint(model.w2 >= 0)
        model.c4 = pmo.constraint(model.x2 <= 1)