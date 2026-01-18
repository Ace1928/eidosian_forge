from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
class GaussianRationalField(GaussianDomain, Field):
    """Field of Gaussian rationals ``QQ_I``

    The :ref:`QQ_I` domain represents the `Gaussian rationals`_ `\\mathbb{Q}(i)`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of rationals and ``I`` (`\\sqrt{-1}`)
    will have the domain :ref:`QQ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I/2)
    >>> p
    Poly(x**2 + I/2, x, domain='QQ_I')
    >>> p.domain
    QQ_I

    The polys option ``gaussian=True`` can be used to specify that the domain
    should be :ref:`QQ_I` even if the coefficients do not contain ``I`` or are
    all integers.

    >>> Poly(x**2)
    Poly(x**2, x, domain='ZZ')
    >>> Poly(x**2 + I)
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> Poly(x**2/2)
    Poly(1/2*x**2, x, domain='QQ')
    >>> Poly(x**2, gaussian=True)
    Poly(x**2, x, domain='QQ_I')
    >>> Poly(x**2 + I, gaussian=True)
    Poly(x**2 + I, x, domain='QQ_I')
    >>> Poly(x**2/2, gaussian=True)
    Poly(1/2*x**2, x, domain='QQ_I')

    The :ref:`QQ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian rationals.

    >>> from sympy import factor, QQ_I
    >>> factor(x**2/4 + 1)
    (x**2 + 4)/4
    >>> factor(x**2/4 + 1, domain='QQ_I')
    (x - 2*I)*(x + 2*I)/4
    >>> factor(x**2/4 + 1, domain=QQ_I)
    (x - 2*I)*(x + 2*I)/4

    It is also possible to specify the :ref:`QQ_I` domain explicitly with
    polys functions like :py:func:`~.apart`.

    >>> from sympy import apart
    >>> apart(1/(1 + x**2))
    1/(x**2 + 1)
    >>> apart(1/(1 + x**2), domain=QQ_I)
    I/(2*(x + I)) - I/(2*(x - I))

    The corresponding `ring of integers`_ is the domain of the Gaussian
    integers :ref:`ZZ_I`. Conversely :ref:`QQ_I` is the `field of fractions`_
    of :ref:`ZZ_I`.

    >>> from sympy import ZZ_I, QQ_I, QQ
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`QQ_I` can be used as a constructor.

    >>> QQ_I(3, 4)
    (3 + 4*I)
    >>> QQ_I(5)
    (5 + 0*I)
    >>> QQ_I(QQ(2, 3), QQ(4, 5))
    (2/3 + 4/5*I)

    The domain elements of :ref:`QQ_I` are instances of
    :py:class:`~.GaussianRational` which support the field operations
    ``+,-,*,**,/``.

    >>> z1 = QQ_I(5, 1)
    >>> z2 = QQ_I(2, QQ(1, 2))
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 1/2*I)
    >>> z1 + z2
    (7 + 3/2*I)
    >>> z1 * z2
    (19/2 + 9/2*I)
    >>> z2 ** 2
    (15/4 + 2*I)

    True division (``/``) in :ref:`QQ_I` gives an element of :ref:`QQ_I` and
    is always exact.

    >>> z1 / z2
    (42/17 + -2/17*I)
    >>> QQ_I.exquo(z1, z2)
    (42/17 + -2/17*I)
    >>> z1 == (z1/z2)*z2
    True

    Both floor (``//``) and modulo (``%``) division can be used with
    :py:class:`~.GaussianRational` (see :py:meth:`~.Domain.div`)
    but division is always exact so there is no remainder.

    >>> z1 // z2
    (42/17 + -2/17*I)
    >>> z1 % z2
    (0 + 0*I)
    >>> QQ_I.div(z1, z2)
    ((42/17 + -2/17*I), (0 + 0*I))
    >>> (z1//z2)*z2 + z1%z2 == z1
    True

    .. _Gaussian rationals: https://en.wikipedia.org/wiki/Gaussian_rational
    """
    dom = QQ
    dtype = GaussianRational
    zero = dtype(QQ(0), QQ(0))
    one = dtype(QQ(1), QQ(0))
    imag_unit = dtype(QQ(0), QQ(1))
    units = (one, imag_unit, -one, -imag_unit)
    rep = 'QQ_I'
    is_GaussianField = True
    is_QQ_I = True

    def __init__(self):
        """For constructing QQ_I."""

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return ZZ_I

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def as_AlgebraicField(self):
        """Get equivalent domain as an ``AlgebraicField``. """
        return AlgebraicField(self.dom, I)

    def numer(self, a):
        """Get the numerator of ``a``."""
        ZZ_I = self.get_ring()
        return ZZ_I.convert(a * self.denom(a))

    def denom(self, a):
        """Get the denominator of ``a``."""
        ZZ = self.dom.get_ring()
        QQ = self.dom
        ZZ_I = self.get_ring()
        denom_ZZ = ZZ.lcm(QQ.denom(a.x), QQ.denom(a.y))
        return ZZ_I(denom_ZZ, ZZ.zero)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to QQ_I."""
        return K1.new(a.x, a.y)

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to QQ_I."""
        return a