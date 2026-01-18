from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import (Symbol, symbols)
from sympy.testing.pytest import XFAIL
def _postprocess_SymbolRemovesOtherSymbols(expr):
    args = tuple((i for i in expr.args if not isinstance(i, Symbol) or isinstance(i, SymbolRemovesOtherSymbols)))
    if args == expr.args:
        return expr
    return Mul.fromiter(args)