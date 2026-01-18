from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
class FinitePoint:
    """
    A point in the upper half space model represented by the quaternion
    z + t * j with t > 0. For example, the point (1 + 2 * i) + 3 * j is::

        sage: from sage.all import *
        sage: FinitePoint(CIF(1,2),RIF(3))
        FinitePoint(1 + 2*I, 3)

    The first argument to :class:`FinitePoint` is z and an element in
    SageMath's ``ComplexIntervalField``. The second argument is t and in
    ``RealIntervalField``.
    """

    def __init__(self, z, t):
        self.z = z
        self.t = t

    def key_interval(self):
        """
        Returns an element in ``RealIntervalField`` which can be used as key
        for an interval tree to implement a mapping from :class:`FinitePoint`::

            sage: from sage.all import *
            sage: FinitePoint(CIF(1,2),RIF(3)).key_interval() # doctest: +NUMERIC12
            36.8919985104477?

        """
        RIF = self.z.real().parent()
        pi = RIF.pi()
        return self.z.real() + self.z.imag() * pi + self.t * pi * pi

    def translate_PSL(self, m):
        """
        Let an extended PSL(2,C)-matrix or a PSL(2,C)-matrix act on the finite
        point.
        The matrix m should be an :class:`ExtendedMatrix` or a SageMath ``Matrix``
        with coefficients in SageMath's ``ComplexIntervalField`` and have
        determinant 1::

            sage: from sage.all import *
            sage: pt = FinitePoint(CIF(1,2),RIF(3))
            sage: m = matrix([[CIF(0.5), CIF(2.4, 2)],[CIF(0.0), CIF(2.0)]])
            sage: pt.translate_PSL(m) # doctest: +NUMERIC12
            FinitePoint(1.4500000000000000? + 1.5000000000000000?*I, 0.75000000000000000?)
            sage: m = ExtendedMatrix(m, isOrientationReversing = True)
            sage: pt.translate_PSL(m) # doctest: +NUMERIC12
            FinitePoint(1.4500000000000000? + 0.50000000000000000?*I, 0.75000000000000000?)

        """
        return self._translate(m, normalize_matrix=False)

    def translate_PGL(self, m):
        """
        Let an extended PGL(2,C)-matrix or a PGL(2,C)-matrix act on the finite
        point.
        The matrix m should be an :class:`ExtendedMatrix` or a SageMath
        ``Matrix`` with coefficients in SageMath's ``ComplexIntervalField``::

            sage: from sage.all import *
            sage: pt = FinitePoint(CIF(1,2),RIF(3))
            sage: m = matrix([[CIF(0.25), CIF(1.2, 1)],[CIF(0.0), CIF(1.0)]])
            sage: pt.translate_PGL(m) # doctest: +NUMERIC12
            FinitePoint(1.4500000000000000? + 1.5000000000000000?*I, 0.75000000000000000?)

        """
        return self._translate(m, normalize_matrix=True)

    def _translate(self, m, normalize_matrix):
        if isinstance(m, ExtendedMatrix):
            mat = m.matrix
            if m.isOrientationReversing:
                z = self.z.conjugate()
            else:
                z = self.z
        else:
            mat = m
            z = self.z
        if normalize_matrix:
            mat = mat / sqrt(mat.det())
        az_b = mat[0, 0] * z + mat[0, 1]
        cz_d = mat[1, 0] * z + mat[1, 1]
        denom = _abs_sqr(cz_d) + _abs_sqr(mat[1, 0] * self.t)
        num = az_b * cz_d.conjugate() + mat[0, 0] * mat[1, 0].conjugate() * self.t ** 2
        return FinitePoint(num / denom, self.t / denom)

    def cosh_dist(self, other):
        """
        Returns cosh of the distance of this finite point to another
        finite point::

            sage: from sage.all import *
            sage: a = FinitePoint(CIF(1,2),RIF(3))
            sage: b = FinitePoint(CIF(4,5),RIF(6))
            sage: a.cosh_dist(b) # doctest: +NUMERIC12
            1.7500000000000000?

        """
        r = 1 + ((self.t - other.t) ** 2 + _abs_sqr(self.z - other.z)) / (2 * self.t * other.t)
        RIF = r.parent()
        if _within_sage:
            if is_RealIntervalFieldElement(r):
                return r.intersection(RIF(1, sage.all.Infinity))
        if r < 1.0:
            return RIF(1.0)
        return r

    def dist(self, other):
        """
        Returns the distance of this finite point to another finite point::

            sage: from sage.all import *
            sage: a = FinitePoint(CIF(1,2),RIF(3))
            sage: b = FinitePoint(CIF(4,5),RIF(6))
            sage: a.dist(b) # doctest: +NUMERIC12
            1.158810360429947?

        """
        return self.cosh_dist(other).arccosh()

    def __repr__(self):
        return 'FinitePoint(%r, %r)' % (self.z, self.t)