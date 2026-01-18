from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable
@public
def PolynomialRing(dom, *gens, **opts):
    """
    Create a generalized multivariate polynomial ring.

    A generalized polynomial ring is defined by a ground field `K`, a set
    of generators (typically `x_1, \\ldots, x_n`) and a monomial order `<`.
    The monomial order can be global, local or mixed. In any case it induces
    a total ordering on the monomials, and there exists for every (non-zero)
    polynomial `f \\in K[x_1, \\ldots, x_n]` a well-defined "leading monomial"
    `LM(f) = LM(f, >)`. One can then define a multiplicative subset
    `S = S_> = \\{f \\in K[x_1, \\ldots, x_n] | LM(f) = 1\\}`. The generalized
    polynomial ring corresponding to the monomial order is
    `R = S^{-1}K[x_1, \\ldots, x_n]`.

    If `>` is a so-called global order, that is `1` is the smallest monomial,
    then we just have `S = K` and `R = K[x_1, \\ldots, x_n]`.

    Examples
    ========

    A few examples may make this clearer.

    >>> from sympy.abc import x, y
    >>> from sympy import QQ

    Our first ring uses global lexicographic order.

    >>> R1 = QQ.old_poly_ring(x, y, order=(("lex", x, y),))

    The second ring uses local lexicographic order. Note that when using a
    single (non-product) order, you can just specify the name and omit the
    variables:

    >>> R2 = QQ.old_poly_ring(x, y, order="ilex")

    The third and fourth rings use a mixed orders:

    >>> o1 = (("ilex", x), ("lex", y))
    >>> o2 = (("lex", x), ("ilex", y))
    >>> R3 = QQ.old_poly_ring(x, y, order=o1)
    >>> R4 = QQ.old_poly_ring(x, y, order=o2)

    We will investigate what elements of `K(x, y)` are contained in the various
    rings.

    >>> L = [x, 1/x, y/(1 + x), 1/(1 + y), 1/(1 + x*y)]
    >>> test = lambda R: [f in R for f in L]

    The first ring is just `K[x, y]`:

    >>> test(R1)
    [True, False, False, False, False]

    The second ring is R1 localised at the maximal ideal (x, y):

    >>> test(R2)
    [True, False, True, True, True]

    The third ring is R1 localised at the prime ideal (x):

    >>> test(R3)
    [True, False, True, False, True]

    Finally the fourth ring is R1 localised at `S = K[x, y] \\setminus yK[y]`:

    >>> test(R4)
    [True, False, False, True, False]
    """
    order = opts.get('order', GeneralizedPolynomialRing.default_order)
    if iterable(order):
        order = build_product_order(order, gens)
    order = monomial_key(order)
    opts['order'] = order
    if order.is_global:
        return GlobalPolynomialRing(dom, *gens, **opts)
    else:
        return GeneralizedPolynomialRing(dom, *gens, **opts)