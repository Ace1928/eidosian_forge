from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def _buchberger(f, ring):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Given a set of multivariate polynomials `F`, finds another
    set `G`, such that Ideal `F = Ideal G` and `G` is a reduced
    Groebner basis.

    The resulting basis is unique and has monic generators if the
    ground domains is a field. Otherwise the result is non-unique
    but Groebner bases over e.g. integers can be computed (if the
    input polynomials are monic).

    Groebner bases can be used to choose specific generators for a
    polynomial ideal. Because these bases are unique you can check
    for ideal equality by comparing the Groebner bases.  To see if
    one polynomial lies in an ideal, divide by the elements in the
    base and see if the remainder vanishes.

    They can also be used to solve systems of polynomial equations
    as,  by choosing lexicographic ordering,  you can eliminate one
    variable at a time, provided that the ideal is zero-dimensional
    (finite number of solutions).

    Notes
    =====

    Algorithm used: an improved version of Buchberger's algorithm
    as presented in T. Becker, V. Weispfenning, Groebner Bases: A
    Computational Approach to Commutative Algebra, Springer, 1993,
    page 232.

    References
    ==========

    .. [1] [Bose03]_
    .. [2] [Giovini91]_
    .. [3] [Ajwa95]_
    .. [4] [Cox97]_

    """
    order = ring.order
    monomial_mul = ring.monomial_mul
    monomial_div = ring.monomial_div
    monomial_lcm = ring.monomial_lcm

    def select(P):
        pr = min(P, key=lambda pair: order(monomial_lcm(f[pair[0]].LM, f[pair[1]].LM)))
        return pr

    def normal(g, J):
        h = g.rem([f[j] for j in J])
        if not h:
            return None
        else:
            h = h.monic()
            if h not in I:
                I[h] = len(f)
                f.append(h)
            return (h.LM, I[h])

    def update(G, B, ih):
        h = f[ih]
        mh = h.LM
        C = G.copy()
        D = set()
        while C:
            ig = C.pop()
            g = f[ig]
            mg = g.LM
            LCMhg = monomial_lcm(mh, mg)

            def lcm_divides(ip):
                m = monomial_lcm(mh, f[ip].LM)
                return monomial_div(LCMhg, m)
            if monomial_mul(mh, mg) == LCMhg or (not any((lcm_divides(ipx) for ipx in C)) and (not any((lcm_divides(pr[1]) for pr in D)))):
                D.add((ih, ig))
        E = set()
        while D:
            ih, ig = D.pop()
            mg = f[ig].LM
            LCMhg = monomial_lcm(mh, mg)
            if not monomial_mul(mh, mg) == LCMhg:
                E.add((ih, ig))
        B_new = set()
        while B:
            ig1, ig2 = B.pop()
            mg1 = f[ig1].LM
            mg2 = f[ig2].LM
            LCM12 = monomial_lcm(mg1, mg2)
            if not monomial_div(LCM12, mh) or monomial_lcm(mg1, mh) == LCM12 or monomial_lcm(mg2, mh) == LCM12:
                B_new.add((ig1, ig2))
        B_new |= E
        G_new = set()
        while G:
            ig = G.pop()
            mg = f[ig].LM
            if not monomial_div(mg, mh):
                G_new.add(ig)
        G_new.add(ih)
        return (G_new, B_new)
    if not f:
        return []
    f1 = f[:]
    while True:
        f = f1[:]
        f1 = []
        for i in range(len(f)):
            p = f[i]
            r = p.rem(f[:i])
            if r:
                f1.append(r.monic())
        if f == f1:
            break
    I = {}
    F = set()
    G = set()
    CP = set()
    for i, h in enumerate(f):
        I[h] = i
        F.add(i)
    while F:
        h = min([f[x] for x in F], key=lambda f: order(f.LM))
        ih = I[h]
        F.remove(ih)
        G, CP = update(G, CP, ih)
    reductions_to_zero = 0
    while CP:
        ig1, ig2 = select(CP)
        CP.remove((ig1, ig2))
        h = spoly(f[ig1], f[ig2], ring)
        G1 = sorted(G, key=lambda g: order(f[g].LM))
        ht = normal(h, G1)
        if ht:
            G, CP = update(G, CP, ht[1])
        else:
            reductions_to_zero += 1
    Gr = set()
    for ig in G:
        ht = normal(f[ig], G - {ig})
        if ht:
            Gr.add(ht[1])
    Gr = [f[ig] for ig in Gr]
    Gr = sorted(Gr, key=lambda f: order(f.LM), reverse=True)
    return Gr