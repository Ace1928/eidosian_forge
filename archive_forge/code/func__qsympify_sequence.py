from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
def _qsympify_sequence(seq):
    """Convert elements of a sequence to standard form.

    This is like sympify, but it performs special logic for arguments passed
    to QExpr. The following conversions are done:

    * (list, tuple, Tuple) => _qsympify_sequence each element and convert
      sequence to a Tuple.
    * basestring => Symbol
    * Matrix => Matrix
    * other => sympify

    Strings are passed to Symbol, not sympify to make sure that variables like
    'pi' are kept as Symbols, not the SymPy built-in number subclasses.

    Examples
    ========

    >>> from sympy.physics.quantum.qexpr import _qsympify_sequence
    >>> _qsympify_sequence((1,2,[3,4,[1,]]))
    (1, 2, (3, 4, (1,)))

    """
    return tuple(__qsympify_sequence_helper(seq))