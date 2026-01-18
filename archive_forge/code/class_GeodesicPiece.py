from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
class GeodesicPiece:
    """
    A line segment in a tetrahedron that can participate in a linked list
    (via prev and next_) to make a loop.

    The line segment is going from endpoints[0] to endpoints[1] of the
    given tetrahedron tet such that endpoints[1] of this piece matches
    endpoints[0] of the next piece (in, probably, a different tetrahedron).

    There is an additional field index that can be used by clients for
    book-keeping purposes, for example, to store the index of the cusp
    obtained by drilling this geodesic.
    """

    def __init__(self, index: Optional[int], tet: Tetrahedron, endpoints: Sequence[Endpoint]):
        self.index: Optional[int] = index
        self.tet: Tetrahedron = tet
        self.endpoints: Sequence[Endpoint] = endpoints
        self.prev = None
        self.next_ = None
        self.tracker = None

    @staticmethod
    def create_and_attach(index: int, tet: Tetrahedron, endpoints: Sequence[Endpoint]):
        """
        Creates a line segment and appends it to tet.geodesic_pieces.
        """
        g = GeodesicPiece(index, tet, endpoints)
        tet.geodesic_pieces.append(g)
        return g

    @staticmethod
    def create_face_to_vertex_and_attach(index: int, tet: Tetrahedron, point: Endpoint, direction: int):
        """
        Creates a line segment between the given endpoint on
        a face and the opposite vertex. If direction is +1,
        the pieces goes from the endpoint to the vertex.
        If direction is -1, it goes the opposite way.

        Also appends the new geodesic piece to tet.geodesic_pieces.
        """
        if point.subsimplex not in simplex.TwoSubsimplices:
            raise ValueError('Expected point to be on a face, but its subsimplex is %d' % point.subsimplex)
        v = simplex.comp(point.subsimplex)
        return GeodesicPiece.create_and_attach(index, tet, [point, Endpoint(tet.R13_vertices[v], v)][::direction])

    @staticmethod
    def make_linked_list(pieces):
        """
        Given a list of pieces, populates next_ and prev of each
        piece to turn it into a linked list.
        """
        n = len(pieces)
        for i in range(n):
            a = pieces[i]
            b = pieces[(i + 1) % n]
            a.next_ = b
            b.prev = a

    def is_face_to_vertex(self) -> bool:
        """
        True if line segment starts on a face and goes to a vertex.
        """
        return self.endpoints[0].subsimplex in simplex.TwoSubsimplices and self.endpoints[1].subsimplex in simplex.ZeroSubsimplices

    @staticmethod
    def replace_by(start_piece, end_piece, pieces) -> None:
        """
        Replaces the pieces between start_piece and end_piece (inclusive)
        by the given (not linked) list of pieces in the linked list that
        start_piece and end_piece participate in.
        """
        if start_piece.prev is end_piece:
            items = pieces + [pieces[0]]
        else:
            items = [start_piece.prev] + pieces + [end_piece.next_]
        for i in range(len(items) - 1):
            a = items[i]
            b = items[i + 1]
            a.next_ = b
            b.prev = a
        for piece in [start_piece, end_piece]:
            if piece.tracker:
                piece.tracker.set_geodesic_piece(pieces[0])
                break

    def __repr__(self):
        return 'GeodesicPiece(%d, %r, %r)' % (self.index, self.tet, self.endpoints)