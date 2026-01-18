import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_block(_BaseTestModel):
    """
    A continuous linear model with nested blocks
    """
    description = 'LP_block'
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.b = Block()
        model.B = Block([1, 2, 3])
        model.a = Param(initialize=1.0, mutable=True)
        model.b.x = Var(within=NonNegativeReals)
        model.B[1].x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.b.x + 3.0 * model.B[1].x)
        model.obj.deactivate()
        model.B[2].c = Constraint(expr=-model.B[1].x <= -model.a)
        model.B[2].obj = Objective(expr=model.b.x + 3.0 * model.B[1].x + 2)
        model.B[3].c = Constraint(expr=(2.0, model.b.x / model.a - model.B[1].x, 10))

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.b.x.value = 1.0
        model.B[1].x.value = 1.0