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
class GeneralizedPolynomialRing(PolynomialRingBase):
    """A generalized polynomial ring, with objects DMF. """
    dtype = DMF

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        res = self.dtype(a, self.dom, len(self.gens) - 1, ring=self)
        if res.denom().terms(order=self.order)[0][0] != (0,) * len(self.gens):
            from sympy.printing.str import sstr
            raise CoercionFailed('denominator %s not allowed in %s' % (sstr(res), self))
        return res

    def __contains__(self, a):
        try:
            a = self.convert(a)
        except CoercionFailed:
            return False
        return a.denom().terms(order=self.order)[0][0] == (0,) * len(self.gens)

    def from_FractionField(K1, a, K0):
        dmf = K1.get_field().from_FractionField(a, K0)
        return K1((dmf.num, dmf.den))

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return basic_from_dict(a.numer().to_sympy_dict(), *self.gens) / basic_from_dict(a.denom().to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        p, q = a.as_numer_denom()
        num, _ = dict_from_basic(p, gens=self.gens)
        den, _ = dict_from_basic(q, gens=self.gens)
        for k, v in num.items():
            num[k] = self.dom.from_sympy(v)
        for k, v in den.items():
            den[k] = self.dom.from_sympy(v)
        return self((num, den)).cancel()

    def _vector_to_sdm(self, v, order):
        """
        Turn an iterable into a sparse distributed module.

        Note that the vector is multiplied by a unit first to make all entries
        polynomials.

        Examples
        ========

        >>> from sympy import ilex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> f = R.convert((x + 2*y) / (1 + x))
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], ilex)
        [((0, 0, 1), 2), ((0, 1, 0), 1), ((1, 1, 1), 1), ((1,
          2, 1), 1)]
        """
        u = self.one.numer()
        for x in v:
            u *= x.denom()
        return _vector_to_sdm_helper([x.numer() * u / x.denom() for x in v], order)