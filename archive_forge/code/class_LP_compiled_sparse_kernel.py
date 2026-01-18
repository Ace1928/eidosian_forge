import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
from pyomo.repn.beta.matrix import compile_block_linear_constraints
@register_model
class LP_compiled_sparse_kernel(LP_compiled_dense_kernel):

    def _generate_model(self):
        x = self._generate_base_model()
        model = self.model
        A, lb, ub, eq_index = self._get_dense_data()
        model.Amatrix = pmo.matrix_constraint(A, lb=lb, ub=ub, x=x, sparse=True)
        for i in eq_index:
            assert model.Amatrix[i].lb == model.Amatrix[i].ub
            model.Amatrix[i].rhs = model.Amatrix[i].lb