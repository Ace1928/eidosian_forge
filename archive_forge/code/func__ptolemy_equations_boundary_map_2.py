from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _ptolemy_equations_boundary_map_2(self):
    """
        This is reimplementing get_ptolemy_equations_boundary_map_2
        from addl_code/ptolemy_equations.c

        Boundary map C_3 -> C_2 in relative cellular homology H_2(M, boundary M)
        represented as matrix

        The following map represents the boundary map in the cellular chain
        complex when representing a linear map as a matrix m acting on a column
        vector v by left-multiplication m * v. With right-multiplication acting
        on row vectors, the matrix represents the coboundary map in the cochain
        complex.

        The basis for C_3 are just the oriented tetrahedra of the triangulation.
        The basis for C_2 are the face classes, see
        _ptolemy_equations_identified_face_classes.
        """

    def process_edge(edge):
        row = [0 for i in range(self.getNumberOfTriangles())]
        for edgeEmbedding in edge.getEmbeddings():
            tet = edgeEmbedding.getTetrahedron()
            perm = edgeEmbedding.getVertices()
            for face in [2, 3]:
                sign = perm.sign() * (-1) ** face * NTriangulationForPtolemy._sign_of_tetrahedron_face(tet, perm[face])
                index = self._index_of_tetrahedron_face(tet, perm[face])
                row[index] += sign
        return [x / 2 for x in row]
    matrix = [process_edge(edge) for edge in self.getEdges()]
    row_explanations = ['edge_%d' % i for i in range(self.getNumberOfEdges())]
    column_explanations = self._face_class_explanations()
    return (matrix, row_explanations, column_explanations)