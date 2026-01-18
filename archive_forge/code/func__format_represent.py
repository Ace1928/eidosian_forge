from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
def _format_represent(self, result, format):
    if format == 'sympy' and (not isinstance(result, Matrix)):
        return to_sympy(result)
    elif format == 'numpy' and (not isinstance(result, numpy_ndarray)):
        return to_numpy(result)
    elif format == 'scipy.sparse' and (not isinstance(result, scipy_sparse_matrix)):
        return to_scipy_sparse(result)
    return result