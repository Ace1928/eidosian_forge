from ... import sage_helper
from .. import t3mlite as t3m
def B2(self):
    """
        The matrix describing the boundary map C_2 -> C_1

        Does *not* assume that the faces are numbered like
        range(len(faces)).
        """
    if self._B2 is None:
        E, F = (len(self.edges), len(self.faces))
        assert list(range(E)) == sorted((e.index for e in self.edges))
        D = matrix(ZZ, E, F, sparse=True)
        for i, face in enumerate(self.faces):
            for edge, sign in face.edges_with_orientations:
                D[edge.index, i] += sign
        self._B2 = D
    return self._B2