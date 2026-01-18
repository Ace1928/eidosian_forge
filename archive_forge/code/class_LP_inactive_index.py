import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_inactive_index(_BaseTestModel):
    """
    A continuous linear model where component subindices have been deactivated
    """
    description = 'LP_inactive_index'
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.s = Set(initialize=[1, 2])
        model.x = Var()
        model.y = Var()
        model.z = Var(bounds=(0, None))
        model.obj = Objective(model.s, rule=inactive_index_LP_obj_rule)
        model.OBJ = Objective(expr=model.x + model.y)
        model.obj[1].deactivate()
        model.OBJ.deactivate()
        model.c1 = ConstraintList()
        model.c1.add(model.x <= 1)
        model.c1.add(model.x >= -1)
        model.c1.add(model.y <= 1)
        model.c1.add(model.y >= -1)
        model.c1[1].deactivate()
        model.c1[4].deactivate()
        model.c2 = Constraint(model.s, rule=inactive_index_LP_c2_rule)
        model.b = Block()
        model.b.c = Constraint(expr=model.z >= 2)
        model.B = Block(model.s)
        model.B[1].c = Constraint(expr=model.z >= 3)
        model.B[2].c = Constraint(expr=model.z >= 1)
        model.b.deactivate()
        model.B.deactivate()
        model.B[2].activate()

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None
        model.z.value = 2.0