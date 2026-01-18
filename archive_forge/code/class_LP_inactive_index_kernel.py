import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_inactive_index_kernel(LP_inactive_index):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.s = [1, 2]
        model.x = pmo.variable()
        model.y = pmo.variable()
        model.z = pmo.variable(lb=0)
        model.obj = pmo.objective_dict()
        for i in model.s:
            model.obj[i] = pmo.objective(inactive_index_LP_obj_rule(model, i))
        model.OBJ = pmo.objective(model.x + model.y)
        model.obj[1].deactivate()
        model.OBJ.deactivate()
        model.c1 = pmo.constraint_dict()
        model.c1[1] = pmo.constraint(model.x <= 1)
        model.c1[2] = pmo.constraint(model.x >= -1)
        model.c1[3] = pmo.constraint(model.y <= 1)
        model.c1[4] = pmo.constraint(model.y >= -1)
        model.c1[1].deactivate()
        model.c1[4].deactivate()
        model.c2 = pmo.constraint_dict()
        for i in model.s:
            model.c2[i] = pmo.constraint(inactive_index_LP_c2_rule(model, i))
        model.b = pmo.block()
        model.b.c = pmo.constraint(model.z >= 2)
        model.B = pmo.block_dict()
        model.B[1] = pmo.block()
        model.B[1].c = pmo.constraint(model.z >= 3)
        model.B[2] = pmo.block()
        model.B[2].c = pmo.constraint(model.z >= 1)
        model.b.deactivate()
        model.B[1].deactivate()