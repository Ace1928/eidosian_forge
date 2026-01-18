import itertools
from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
def _print_meijerg(self, e):
    v = {}
    v[0, 0] = [self._print(a) for a in e.an]
    v[0, 1] = [self._print(a) for a in e.aother]
    v[1, 0] = [self._print(b) for b in e.bm]
    v[1, 1] = [self._print(b) for b in e.bother]
    P = self._print(e.argument)
    P.baseline = P.height() // 2
    vp = {}
    for idx in v:
        vp[idx] = self._hprint_vec(v[idx])
    for i in range(2):
        maxw = max(vp[0, i].width(), vp[1, i].width())
        for j in range(2):
            s = vp[j, i]
            left = (maxw - s.width()) // 2
            right = maxw - left - s.width()
            s = prettyForm(*s.left(' ' * left))
            s = prettyForm(*s.right(' ' * right))
            vp[j, i] = s
    D1 = prettyForm(*vp[0, 0].right('  ', vp[0, 1]))
    D1 = prettyForm(*D1.below(' '))
    D2 = prettyForm(*vp[1, 0].right('  ', vp[1, 1]))
    D = prettyForm(*D1.below(D2))
    D.baseline = D.height() // 2
    P = prettyForm(*P.left(' '))
    D = prettyForm(*D.right(' '))
    D = self._hprint_vseparator(D, P)
    D = prettyForm(*D.parens('(', ')'))
    above = D.height() // 2 - 1
    below = D.height() - above - 1
    sz, t, b, add, img = annotated('G')
    F = prettyForm('\n' * (above - t) + img + '\n' * (below - b), baseline=above + sz)
    pp = self._print(len(e.ap))
    pq = self._print(len(e.bq))
    pm = self._print(len(e.bm))
    pn = self._print(len(e.an))

    def adjust(p1, p2):
        diff = p1.width() - p2.width()
        if diff == 0:
            return (p1, p2)
        elif diff > 0:
            return (p1, prettyForm(*p2.left(' ' * diff)))
        else:
            return (prettyForm(*p1.left(' ' * -diff)), p2)
    pp, pm = adjust(pp, pm)
    pq, pn = adjust(pq, pn)
    pu = prettyForm(*pm.right(', ', pn))
    pl = prettyForm(*pp.right(', ', pq))
    ht = F.baseline - above - 2
    if ht > 0:
        pu = prettyForm(*pu.below('\n' * ht))
    p = prettyForm(*pu.below(pl))
    F.baseline = above
    F = prettyForm(*F.right(p))
    F.baseline = above + add
    D = prettyForm(*F.right(' ', D))
    return D