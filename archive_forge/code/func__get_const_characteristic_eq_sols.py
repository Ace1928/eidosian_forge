from collections import defaultdict
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, \
from sympy.integrals import Integral
from sympy.polys import (Poly, RootOf, rootof, roots)
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp # type: ignore
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants
def _get_const_characteristic_eq_sols(r, func, order):
    """
    Returns the roots of characteristic equation of constant coefficient
    linear ODE and list of collectterms which is later on used by simplification
    to use collect on solution.

    The parameter `r` is a dict of order:coeff terms, where order is the order of the
    derivative on each term, and coeff is the coefficient of that derivative.

    """
    x = func.args[0]
    chareq, symbol = (S.Zero, Dummy('x'))
    for i in r.keys():
        if isinstance(i, str) or i < 0:
            pass
        else:
            chareq += r[i] * symbol ** i
    chareq = Poly(chareq, symbol)
    chareqroots = roots(chareq, multiple=True)
    if len(chareqroots) != order:
        chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]
    chareq_is_complex = not all((i.is_real for i in chareq.all_coeffs()))
    charroots = defaultdict(int)
    for root in chareqroots:
        charroots[root] += 1
    collectterms = []
    gensols = []
    conjugate_roots = []
    for root in chareqroots:
        if root not in charroots:
            continue
        multiplicity = charroots.pop(root)
        for i in range(multiplicity):
            if chareq_is_complex:
                gensols.append(x ** i * exp(root * x))
                collectterms = [(i, root, 0)] + collectterms
                continue
            reroot = re(root)
            imroot = im(root)
            if imroot.has(atan2) and reroot.has(atan2):
                gensols.append(x ** i * exp(root * x))
                collectterms = [(i, root, 0)] + collectterms
            else:
                if root in conjugate_roots:
                    collectterms = [(i, reroot, imroot)] + collectterms
                    continue
                if imroot == 0:
                    gensols.append(x ** i * exp(reroot * x))
                    collectterms = [(i, reroot, 0)] + collectterms
                    continue
                conjugate_roots.append(conjugate(root))
                gensols.append(x ** i * exp(reroot * x) * sin(abs(imroot) * x))
                gensols.append(x ** i * exp(reroot * x) * cos(imroot * x))
                collectterms = [(i, reroot, imroot)] + collectterms
    return (gensols, collectterms)