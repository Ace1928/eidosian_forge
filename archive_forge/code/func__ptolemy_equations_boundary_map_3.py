from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _ptolemy_equations_boundary_map_3(self):
    """
        This is reimplementing get_ptolemy_equations_boundary_map_3
        from addl_code/ptolemy_equations.c

        Boundary map C_2 -> C_1 in cellular homology represented as matrix.

        Also see _ptolemy_equations_boundary_map_2.
        """

    def process_triangle(triangle):
        row = [0 for i in range(self.getNumberOfTetrahedra())]
        for i in range(2):
            index = self.tetrahedronIndex(triangle.getEmbedding(i).getTetrahedron())
            row[index] += (-1) ** i
        return row
    matrix = [process_triangle(triangle) for triangle in self.getTriangles()]
    row_explanations = self._face_class_explanations()
    column_explanations = ['tet_%d' % i for i in range(self.getNumberOfTetrahedra())]
    return (matrix, row_explanations, column_explanations)