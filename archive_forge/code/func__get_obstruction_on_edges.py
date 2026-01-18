from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _get_obstruction_on_edges(self, obstruction_class, tet, face, N):
    """
        This reimplements _get_obstruction_on_edges from
        addl_code/ptolemy_equations.c
        """
    v0 = (face + 1) % 4
    v1 = (face + 2) % 4
    v2 = (face + 3) % 4
    e01 = self._get_obstruction_on_edge_with_other_tet(obstruction_class, tet, face, v0, v1)
    e02 = self._get_obstruction_on_edge_with_other_tet(obstruction_class, tet, face, v0, v2)
    e12 = self._get_obstruction_on_edge_with_other_tet(obstruction_class, tet, face, v1, v2)
    assert (e01 + e12 - e02) % N == 0
    return (e01, e02)