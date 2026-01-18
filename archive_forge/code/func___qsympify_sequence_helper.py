from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
def __qsympify_sequence_helper(seq):
    """
       Helper function for _qsympify_sequence
       This function does the actual work.
    """
    if not is_sequence(seq):
        if isinstance(seq, Matrix):
            return seq
        elif isinstance(seq, str):
            return Symbol(seq)
        else:
            return sympify(seq)
    if isinstance(seq, QExpr):
        return seq
    result = [__qsympify_sequence_helper(item) for item in seq]
    return Tuple(*result)