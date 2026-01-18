from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
@staticmethod
def _sign_of_tetrahedron_face(tetrahedron, face):
    """
        Recall the comments from _index_of_tetrahedron_face

        A regina triangle has two embeddings and thus yields
        two (tetrahedron, face) pairs representing a face class,
        albeit with different signs.
        We assume that the first embedding always represents the
        + face class and the second one - face class.

        In addl_code/ptolemy_equations.c, this was done in
        _fill_tet_face_to_index_data . There, the choice was made
        that the pair (tetrahedron, face) with the lower tetrahedron
        index (face index in case tie) represents + face class.
        """
    triangle = tetrahedron.getTriangle(face)
    embedding = triangle.getEmbedding(0)
    if tetrahedron == embedding.getTetrahedron() and embedding.getFace() == face:
        return +1
    else:
        return -1