from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import oo, equal_valued
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.printing.pretty.stringpict import stringPict
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
def _eval_innerproduct(self, bra, **hints):
    if len(self.args) != len(bra.args):
        raise ValueError('Cannot multiply a ket that has a different number of labels.')
    for arg, bra_arg in zip(self.args, bra.args):
        diff = arg - bra_arg
        diff = diff.expand()
        is_zero = diff.is_zero
        if is_zero is False:
            return S.Zero
        if is_zero is None:
            return None
    return S.One