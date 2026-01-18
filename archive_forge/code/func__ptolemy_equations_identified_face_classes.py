from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _ptolemy_equations_identified_face_classes(self):
    """
        This is reimplementing get_ptolemy_equations_identified_face_classes
        from addl_code/ptolemy_equations.c

        This function returns an identification structure where s_f_t gets
        identified with -s_g_u if face f of tetrahedron t is glued to face g of
        tetrahedron u.

        We can represent a 2-cohomology class H^2(M,boundary M) by denoting by
        s_f_t the value the 2-cohomology class takes on the face f of
        tetrahedron t with the orientation being the one induced from the
        orientation of the tetrahedron.
        Because a face class of the triangulation has two representatives
        (tet_index, face_index) and the gluing is orientation-reversing on the
        face, one s will be the negative of another s.
        """

    def process_embedding(embedding):
        face = embedding.getFace()
        tet = self.tetrahedronIndex(embedding.getTetrahedron())
        return 's_%d_%d' % (face, tet)

    def process_triangle(triangle):
        return (-1, 0, process_embedding(triangle.getEmbedding(0)), process_embedding(triangle.getEmbedding(1)))
    return [process_triangle(triangle) for triangle in self.getTriangles()]