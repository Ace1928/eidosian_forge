from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
class GaussianDomain:
    """Base class for Gaussian domains."""
    dom = None
    is_Numerical = True
    is_Exact = True
    has_assoc_Ring = True
    has_assoc_Field = True

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        conv = self.dom.to_sympy
        return conv(a.x) + I * conv(a.y)

    def from_sympy(self, a):
        """Convert a SymPy object to ``self.dtype``."""
        r, b = a.as_coeff_Add()
        x = self.dom.from_sympy(r)
        if not b:
            return self.new(x, 0)
        r, b = b.as_coeff_Mul()
        y = self.dom.from_sympy(r)
        if b is I:
            return self.new(x, y)
        else:
            raise CoercionFailed('{} is not Gaussian'.format(a))

    def inject(self, *gens):
        """Inject generators into this domain. """
        return self.poly_ring(*gens)

    def canonical_unit(self, d):
        unit = self.units[-d.quadrant()]
        return unit

    def is_negative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_positive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonnegative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY mpz to ``self.dtype``."""
        return K1(a)

    def from_ZZ(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_QQ(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_python(K1, a, K0):
        """Convert a QQ_python element to ``self.dtype``."""
        return K1(a)

    def from_AlgebraicField(K1, a, K0):
        """Convert an element from ZZ<I> or QQ<I> to ``self.dtype``."""
        if K0.ext.args[0] == I:
            return K1.from_sympy(K0.to_sympy(a))