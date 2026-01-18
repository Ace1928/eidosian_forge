import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
from pyomo.repn.beta.matrix import compile_block_linear_constraints
@register_model
class LP_compiled_dense_kernel(LP_compiled):

    def _get_dense_data(self):
        assert has_numpy and has_scipy
        A = numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        lb = numpy.array([-1.0, -numpy.inf, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -2.0, -1.0, -numpy.inf, 0.0, -2.0, -3.0, -1.0, -numpy.inf, 0.0, -2.0, -numpy.inf])
        ub = numpy.array([numpy.inf, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 1.0, numpy.inf, 1.0, 0.0, 1.0, 0.0, numpy.inf, 0.0, 0.0, numpy.inf, 2.0])
        eq_index = [2, 3, 4, 5, 14, 19]
        return (A, lb, ub, eq_index)

    def _generate_base_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.s = list(range(1, 13))
        model.x = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
        model.x[1].lb = -1
        model.x[1].ub = 1
        model.x[2].lb = -1
        model.x[2].ub = 1
        model.obj = pmo.objective(expr=sum((model.x[i] * (-1) ** (i + 1) for i in model.s)))
        variable_order = [model.x[3], model.x[4], model.x[5], model.x[6], model.x[7], model.x[8], model.x[9], model.x[10], model.x[11], model.x[12]]
        return variable_order

    def _generate_model(self):
        x = self._generate_base_model()
        model = self.model
        A, lb, ub, eq_index = self._get_dense_data()
        model.Amatrix = pmo.matrix_constraint(A, lb=lb, ub=ub, x=x, sparse=False)
        for i in eq_index:
            assert model.Amatrix[i].lb == model.Amatrix[i].ub
            model.Amatrix[i].rhs = model.Amatrix[i].lb