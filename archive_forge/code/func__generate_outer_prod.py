from itertools import product
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy
from sympy.physics.quantum.tensorproduct import TensorProduct, tensor_product_simp
from sympy.physics.quantum.trace import Tr
def _generate_outer_prod(self, arg1, arg2):
    c_part1, nc_part1 = arg1.args_cnc()
    c_part2, nc_part2 = arg2.args_cnc()
    if len(nc_part1) == 0 or len(nc_part2) == 0:
        raise ValueError('Atleast one-pair of Non-commutative instance required for outer product.')
    if isinstance(nc_part1[0], TensorProduct) and len(nc_part1) == 1 and (len(nc_part2) == 1):
        op = tensor_product_simp(nc_part1[0] * Dagger(nc_part2[0]))
    else:
        op = Mul(*nc_part1) * Dagger(Mul(*nc_part2))
    return Mul(*c_part1) * Mul(*c_part2) * op