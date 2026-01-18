from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
def _print_sequence_pretty(self, seq, sep, printer, *args):
    pform = printer._print(seq[0], *args)
    for item in seq[1:]:
        pform = prettyForm(*pform.right(sep))
        pform = prettyForm(*pform.right(printer._print(item, *args)))
    return pform