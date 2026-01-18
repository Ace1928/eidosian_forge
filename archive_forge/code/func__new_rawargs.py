from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
@classmethod
def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
    """Create new instance of this class with hilbert_space and args.

        This is used to bypass the more complex logic in the ``__new__``
        method in cases where you already have the exact ``hilbert_space``
        and ``args``. This should be used when you are positive these
        arguments are valid, in their final, proper form and want to optimize
        the creation of the object.
        """
    obj = Expr.__new__(cls, *args, **old_assumptions)
    obj.hilbert_space = hilbert_space
    return obj