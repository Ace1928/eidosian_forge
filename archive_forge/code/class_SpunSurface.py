import snappy
import FXrays
class SpunSurface:
    """
    A spun normal surface in an ideal triangulation, as introduced by
    Thurston.

    For an quick sketch of this theory see `[DG]
    <http://arxiv.org/abs/1102.4588>`_ and for more details see
    `[Tillmann] <http://arxiv.org/abs/math/0406271>`_.

    The quad conventions are (Q02, Q03, Q01) corresponding to
    z -> 0, z' -> 0, and z'' -> 0 respectively, as per Figure 3.1 of
    `[DG] <http://arxiv.org/abs/1102.4588>`_.  The quad types
    are numbered 0, 1, 2; the "None" quad type means a
    tetrahedron contains no quads at all.
    """

    def __init__(self, manifold, quad_vector=None, quad_types=None, index=None):
        self._manifold = manifold
        self._index = index
        if quad_types is not None:
            coefficients = quad_vector
            quad_vector = []
            for c, q in zip(coefficients, quad_types):
                three = [0, 0, 0]
                three[q] = c
                quad_vector += three
        quad_vector = Vector(quad_vector)
        eqns = manifold._normal_surface_equations()
        assert eqns.is_solution(quad_vector)
        self._quad_vector = quad_vector
        self._quad_types, self._coefficients = quad_vector_to_type_and_coeffs(quad_vector)
        self._boundary_slopes = eqns.boundary_slope_of_solution(quad_vector)

    def quad_vector(self):
        return self._quad_vector

    def quad_types(self):
        return self._quad_types

    def coefficients(self):
        return self._coefficients

    def boundary_slopes(self):
        return self._boundary_slopes

    def is_compatible(self, other):
        for a, b in zip(self._quad_types, other._quad_types):
            if not (a == b or None in (a, b)):
                return False
        return True

    def __radd__(self, other):
        if other == 0:
            return self

    def __add__(self, other):
        if other == 0:
            return self
        if not self.is_compatible(other):
            raise ValueError('Normal surfaces are not compatible')
        return SpunSurface(self._manifold, self._quad_vector + other._quad_vector)

    def __repr__(self):
        return '<Surface %s: %s %s %s>' % (self._index, self._quad_types, list(self._coefficients), tuple(self._boundary_slopes))