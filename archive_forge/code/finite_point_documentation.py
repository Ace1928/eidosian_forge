from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix

        Returns the distance of this finite point to another finite point::

            sage: from sage.all import *
            sage: a = FinitePoint(CIF(1,2),RIF(3))
            sage: b = FinitePoint(CIF(4,5),RIF(6))
            sage: a.dist(b) # doctest: +NUMERIC12
            1.158810360429947?

        