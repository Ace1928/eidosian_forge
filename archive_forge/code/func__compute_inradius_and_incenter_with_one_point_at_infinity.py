from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def _compute_inradius_and_incenter_with_one_point_at_infinity(nonInfPoints):
    """
    Computes inradius and incenter for a tetrahedron spanned by infinity and
    the given three ideal points.
    """
    if not len(nonInfPoints) == 3:
        raise Exception('Expects three non-infinite points.')
    lengths = [abs(nonInfPoints[(i + 2) % 3] - nonInfPoints[(i + 1) % 3]) for i in range(3)]
    length_total = sum(lengths)
    length_product = lengths[0] * lengths[1] * lengths[2]
    terms_product = (-lengths[0] + lengths[1] + lengths[2]) * (lengths[0] - lengths[1] + lengths[2]) * (lengths[0] + lengths[1] - lengths[2])
    inRadiusSqr = terms_product / length_total / 4
    inRadius = inRadiusSqr.sqrt()
    circumRadius = length_product / (terms_product * length_total).sqrt()
    eHeightSqr = inRadiusSqr + 4 * inRadius * circumRadius
    eHeight = eHeightSqr.sqrt()
    height = (eHeightSqr - inRadiusSqr).sqrt()
    radius = ((eHeight + inRadius) / (eHeight - inRadius)).log() / 2
    incenter = sum([pt * l for pt, l in zip(nonInfPoints, lengths)]) / length_total
    return (radius, FinitePoint(incenter, height))