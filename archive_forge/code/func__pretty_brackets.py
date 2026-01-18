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
def _pretty_brackets(self, height, use_unicode=True):
    if use_unicode:
        lbracket, rbracket = (getattr(self, 'lbracket_ucode', ''), getattr(self, 'rbracket_ucode', ''))
        slash, bslash, vert = ('╱', '╲', '│')
    else:
        lbracket, rbracket = (getattr(self, 'lbracket', ''), getattr(self, 'rbracket', ''))
        slash, bslash, vert = ('/', '\\', '|')
    if height == 1:
        return (stringPict(lbracket), stringPict(rbracket))
    height += height % 2
    brackets = []
    for bracket in (lbracket, rbracket):
        if bracket in {_lbracket, _lbracket_ucode}:
            bracket_args = [' ' * (height // 2 - i - 1) + slash for i in range(height // 2)]
            bracket_args.extend([' ' * i + bslash for i in range(height // 2)])
        elif bracket in {_rbracket, _rbracket_ucode}:
            bracket_args = [' ' * i + bslash for i in range(height // 2)]
            bracket_args.extend([' ' * (height // 2 - i - 1) + slash for i in range(height // 2)])
        elif bracket in {_straight_bracket, _straight_bracket_ucode}:
            bracket_args = [vert] * height
        else:
            raise ValueError(bracket)
        brackets.append(stringPict('\n'.join(bracket_args), baseline=height // 2))
    return brackets