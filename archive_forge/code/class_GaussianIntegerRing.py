from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
class GaussianIntegerRing(GaussianDomain, Ring):
    """Ring of Gaussian integers ``ZZ_I``

    The :ref:`ZZ_I` domain represents the `Gaussian integers`_ `\\mathbb{Z}[i]`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of integers and ``I`` (`\\sqrt{-1}`)
    will have the domain :ref:`ZZ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I)
    >>> p
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> p.domain
    ZZ_I

    The :ref:`ZZ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian integers.

    >>> from sympy import factor
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, domain='ZZ_I')
    (x - I)*(x + I)

    The corresponding `field of fractions`_ is the domain of the Gaussian
    rationals :ref:`QQ_I`. Conversely :ref:`ZZ_I` is the `ring of integers`_
    of :ref:`QQ_I`.

    >>> from sympy import ZZ_I, QQ_I
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`ZZ_I` can be used as a constructor.

    >>> ZZ_I(3, 4)
    (3 + 4*I)
    >>> ZZ_I(5)
    (5 + 0*I)

    The domain elements of :ref:`ZZ_I` are instances of
    :py:class:`~.GaussianInteger` which support the rings operations
    ``+,-,*,**``.

    >>> z1 = ZZ_I(5, 1)
    >>> z2 = ZZ_I(2, 3)
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 3*I)
    >>> z1 + z2
    (7 + 4*I)
    >>> z1 * z2
    (7 + 17*I)
    >>> z1 ** 2
    (24 + 10*I)

    Both floor (``//``) and modulo (``%``) division work with
    :py:class:`~.GaussianInteger` (see the :py:meth:`~.Domain.div` method).

    >>> z3, z4 = ZZ_I(5), ZZ_I(1, 3)
    >>> z3 // z4  # floor division
    (1 + -1*I)
    >>> z3 % z4   # modulo division (remainder)
    (1 + -2*I)
    >>> (z3//z4)*z4 + z3%z4 == z3
    True

    True division (``/``) in :ref:`ZZ_I` gives an element of :ref:`QQ_I`. The
    :py:meth:`~.Domain.exquo` method can be used to divide in :ref:`ZZ_I` when
    exact division is possible.

    >>> z1 / z2
    (1 + -1*I)
    >>> ZZ_I.exquo(z1, z2)
    (1 + -1*I)
    >>> z3 / z4
    (1/2 + -3/2*I)
    >>> ZZ_I.exquo(z3, z4)
    Traceback (most recent call last):
        ...
    ExactQuotientFailed: (1 + 3*I) does not divide (5 + 0*I) in ZZ_I

    The :py:meth:`~.Domain.gcd` method can be used to compute the `gcd`_ of any
    two elements.

    >>> ZZ_I.gcd(ZZ_I(10), ZZ_I(2))
    (2 + 0*I)
    >>> ZZ_I.gcd(ZZ_I(5), ZZ_I(2, 1))
    (2 + 1*I)

    .. _Gaussian integers: https://en.wikipedia.org/wiki/Gaussian_integer
    .. _gcd: https://en.wikipedia.org/wiki/Greatest_common_divisor

    """
    dom = ZZ
    dtype = GaussianInteger
    zero = dtype(ZZ(0), ZZ(0))
    one = dtype(ZZ(1), ZZ(0))
    imag_unit = dtype(ZZ(0), ZZ(1))
    units = (one, imag_unit, -one, -imag_unit)
    rep = 'ZZ_I'
    is_GaussianRing = True
    is_ZZ_I = True

    def __init__(self):
        """For constructing ZZ_I."""

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_field(self):
        """Returns a field associated with ``self``. """
        return QQ_I

    def normalize(self, d, *args):
        """Return first quadrant element associated with ``d``.

        Also multiply the other arguments by the same power of i.
        """
        unit = self.canonical_unit(d)
        d *= unit
        args = tuple((a * unit for a in args))
        return (d,) + args if args else d

    def gcd(self, a, b):
        """Greatest common divisor of a and b over ZZ_I."""
        while b:
            a, b = (b, a % b)
        return self.normalize(a)

    def lcm(self, a, b):
        """Least common multiple of a and b over ZZ_I."""
        return a * b // self.gcd(a, b)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to ZZ_I."""
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to ZZ_I."""
        return K1.new(ZZ.convert(a.x), ZZ.convert(a.y))