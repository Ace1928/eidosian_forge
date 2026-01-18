from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.tensor.array import NDimArray
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.utilities.iterables import has_variety, sift
import re
def _print_Poly(self, poly):
    cls = poly.__class__.__name__
    terms = []
    for monom, coeff in poly.terms():
        s_monom = ''
        for i, exp in enumerate(monom):
            if exp > 0:
                if exp == 1:
                    s_monom += self._print(poly.gens[i])
                else:
                    s_monom += self._print(pow(poly.gens[i], exp))
        if coeff.is_Add:
            if s_monom:
                s_coeff = '\\left(%s\\right)' % self._print(coeff)
            else:
                s_coeff = self._print(coeff)
        else:
            if s_monom:
                if coeff is S.One:
                    terms.extend(['+', s_monom])
                    continue
                if coeff is S.NegativeOne:
                    terms.extend(['-', s_monom])
                    continue
            s_coeff = self._print(coeff)
        if not s_monom:
            s_term = s_coeff
        else:
            s_term = s_coeff + ' ' + s_monom
        if s_term.startswith('-'):
            terms.extend(['-', s_term[1:]])
        else:
            terms.extend(['+', s_term])
    if terms[0] in ('-', '+'):
        modifier = terms.pop(0)
        if modifier == '-':
            terms[0] = '-' + terms[0]
    expr = ' '.join(terms)
    gens = list(map(self._print, poly.gens))
    domain = 'domain=%s' % self._print(poly.get_domain())
    args = ', '.join([expr] + gens + [domain])
    if cls in accepted_latex_functions:
        tex = '\\%s {\\left(%s \\right)}' % (cls, args)
    else:
        tex = '\\operatorname{%s}{\\left( %s \\right)}' % (cls, args)
    return tex