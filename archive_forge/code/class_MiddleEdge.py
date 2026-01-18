from . import matrix
class MiddleEdge(Edge):
    """
    A middle edge of a doubly truncated simplex.
    """

    def end_point(self):
        """
        The vertex at the end of the edge.
        """
        tet, v0, v1, v2 = self._start_point
        return Vertex(tet, v0, v2, v1)