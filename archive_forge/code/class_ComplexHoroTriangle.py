from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
class ComplexHoroTriangle:
    """
    A horosphere cross section in the corner of an ideal tetrahedron.
    The sides of the triangle correspond to faces of the tetrahedron.
    The lengths stored for the triangle are complex.
    """

    def __init__(self, tet, vertex, known_side, length_of_side):
        left_side, center_side, right_side, z_left, z_right = HoroTriangleBase._sides_and_cross_ratios(tet, vertex, known_side)
        L = length_of_side
        self.lengths = {center_side: L, left_side: -z_left * L, right_side: -L / z_right}
        absL = abs(L)
        self.area = absL * absL * z_left.imag() / 2
        self._real_lengths_cache = None

    def get_real_lengths(self):
        if not self._real_lengths_cache:
            self._real_lengths_cache = {side: abs(length) for side, length in self.lengths.items()}
        return self._real_lengths_cache

    def rescale(self, t):
        """Rescales the triangle by a Euclidean dilation"""
        for face in self.lengths:
            self.lengths[face] *= t
        self.area *= t * t

    @staticmethod
    def direction_sign():
        return -1

    def add_vertex_positions(self, vertex, edge, position):
        """
        Adds a dictionary vertex_positions mapping
        an edge (such as t3m.simplex.E01) to complex position
        for the vertex of the horotriangle obtained by
        intersecting the edge with the horosphere.

        Two of these positions are computed from the one given
        using the complex edge lengths. The given vertex and
        edge are t3m-style.
        """
        self.vertex_positions = {}
        vertex_link = _face_edge_face_triples_for_vertex_link[vertex]
        for i in range(3):
            if edge == vertex_link[i][1]:
                break
        for j in range(3):
            face0, edge, face1 = vertex_link[(i + j) % 3]
            self.vertex_positions[edge] = position
            position += self.lengths[face1]

    def lift_vertex_positions(self, lifted_position):
        """
        Lift the vertex positions of this triangle. lifted_position is
        used as a guide what branch of the logarithm to use.

        The lifted position is computed as the log of the vertex
        position where it is assumed that the fixed point of the
        holonomy is at the origin.  The branch of the logarithm
        closest to lifted_position is used.
        """
        NumericalField = lifted_position.parent()
        twoPi = 2 * NumericalField.pi()
        I = NumericalField(1j)

        def adjust_log(z):
            logZ = log(z)
            return logZ + ((lifted_position - logZ) / twoPi).imag().round() * twoPi * I
        self.lifted_vertex_positions = {edge: adjust_log(position) for edge, position in self.vertex_positions.items()}