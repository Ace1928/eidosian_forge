from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
class TetrahedronEmbedding:
    """
    A map from a tetrahedron with PL arcs in barycentric coordinates
    into R^3. The map is described by choosing where the vertices of
    the given arrow go, in the standard order::

      (tail, head, opp_tail, opp_head)

    The optional boundary information is for recording which faces
    (if any) of the tetrahedron correspond particular faces of some
    larger convex polytope.
    """

    def __init__(self, arrow, vertex_images, bdry_map=None):
        opp_arrow = arrow.copy().opposite()
        to_arrow = {arrow.tail(): 0, arrow.head(): 1, opp_arrow.tail(): 2, opp_arrow.head(): 3}
        self.vertex_images = [vertex_images[to_arrow[V]] for V in ZeroSubsimplices]
        if bdry_map is not None:
            bdry_map = {i: bdry_map[to_arrow[V]] for i, V in enumerate(ZeroSubsimplices)}
        self.bdry_map = bdry_map
        assert [len(v) for v in vertex_images] == 4 * [3]
        R4_images = [list(v) + [1] for v in self.vertex_images]
        self.matrix = Matrix(R4_images).transpose()
        self.inverse_matrix = self.matrix.inverse()

    def transfer_arcs_to_R3(self, arcs):
        return [arc.transform_to_R3(self.matrix, bdry_map=self.bdry_map) for arc in arcs]

    def transfer_arcs_from_R3(self, arcs):
        return [arc.transform_to_R4(self.inverse_matrix) for arc in arcs]

    def info(self):
        self.tetrahedron.info()
        print(self.matrix)