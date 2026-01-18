from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _ptolemy_equations_identified_coordinates(self, N, obstruction_class=None):
    identifications = []
    for triangle in self.getTriangles():
        embedding = triangle.getEmbedding(0)
        tet = embedding.getTetrahedron()
        tet_index = self.tetrahedronIndex(tet)
        face = embedding.getFace()
        gluing = tet.getAdjacentTetrahedronGluing(face)
        other_tet = tet.getAdjacentTetrahedron(face)
        other_tet_index = self.tetrahedronIndex(other_tet)
        other_gluing = gluing.inverse()
        if obstruction_class:
            e01, e02 = self._get_obstruction_on_edges(obstruction_class, tet, face, N)
        for triple in utilities.triples_with_fixed_sum_iterator(N, skipVertices=True):
            ptolemy_index = triple[0:face] + (0,) + triple[face:]
            other_ptolemy_index = tuple([ptolemy_index[other_gluing[v]] for v in range(4)])
            sign = NTriangulationForPtolemy._compute_sign(ptolemy_index, gluing)
            if obstruction_class:
                power = NTriangulationForPtolemy._get_power_from_obstruction_class(face, e01, e02, ptolemy_index)
            else:
                power = 0
            ptolemy = 'c_%d%d%d%d' % ptolemy_index + '_%d' % tet_index
            other_ptolemy = 'c_%d%d%d%d' % other_ptolemy_index + '_%d' % other_tet_index
            identifications.append((sign, power, ptolemy, other_ptolemy))
    return identifications