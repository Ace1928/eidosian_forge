import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class MILP_simple(_BaseTestModel):
    """
    A simple mixed-integer linear program
    """
    description = 'MILP_simple'
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'integer'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=Binary)
        model.obj = Objective(expr=model.x + 3.0 * model.y)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=(2.0, model.x / model.a - model.y, 10))

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 0.1
        model.y.value = 0