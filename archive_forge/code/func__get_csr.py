import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base.matrix_constraint import MatrixConstraint
def _get_csr(m, n, value):
    data = [value] * (m * n)
    indices = [j for j in range(n) for i in range(m)]
    indptr = [0]
    for i in range(m):
        indptr.append(indptr[-1] + n)
    return (data, indices, indptr)