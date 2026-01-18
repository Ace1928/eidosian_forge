from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
@staticmethod
def _get_power_from_obstruction_class(face, e01, e02, ptolemy_index):
    """
        This reimplements _get_power_from_obstruction_class
        from addl_code/ptolemy_equations.c

        Let face face of a tetrahedron be glued to some other face. This
        will identify two Ptolemy coordinates up to a sign and an N-th root
        of unity (for an PSL(N,C)-representation. I.e., we get an equation
        between Ptolemy coordinates of the form
        (+/-) u^p c_index_t = c_index'_t'
        where u is the N-th root of unity, c_index_t is the Ptolemy coordinate
        on the given face and c_index'_t' on the face that it glued to the
        given face.
        _compute_sign will give the sign (+/-) based on the index and the
        face gluing permutation.
        _get_power_from_obstruction_class will give p given the face, the
        Ptolemy index and the edge cocycle that assigns e01 to the edge 01
        and e02 to the edge e02 that is determined through the cohomology
        obstruction class by _get_obstruction_on_edges.
        """
    v1 = (face + 2) % 4
    v2 = (face + 3) % 4
    return ptolemy_index[v1] * e01 + ptolemy_index[v2] * e02