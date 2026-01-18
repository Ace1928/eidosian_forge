from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _get_obstruction_on_edge_with_other_tet(self, obstruction_class, tet, face, v0, v1):
    """
        Reimplements _get_obstruction_on_edge_with_other_tet from
        addl_code/ptolemy_equations.c
        """
    other_tet = tet.getAdjacentTetrahedron(face)
    gluing = tet.getAdjacentTetrahedronGluing(face)
    other_v0 = gluing[v0]
    other_v1 = gluing[v1]
    return self._get_obstruction_on_edge(obstruction_class, tet, v0, v1) - self._get_obstruction_on_edge(obstruction_class, other_tet, other_v0, other_v1)