from ..sage_helper import sage_method, _within_sage
from ..number import Number
from . import verifyHyperbolicity
def _volume_from_shape(z):
    """
    Computes the Bloch-Wigner dilogarithm for z which gives the volume of a
    tetrahedron of the given shape.
    """
    if _within_sage:
        CIF = z.parent()
        if is_ComplexIntervalField(CIF):
            CBF = ComplexBallField(CIF.precision())
            RIF = RealIntervalField(CIF.precision())
            return RIF(_unprotected_volume_from_shape(CBF(z)))
        else:
            z = Number(z)
    return z.volume()