from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
class TetrahedronEmbeddingCache:

    def __init__(self):
        self.cache = dict()

    def __call__(self, arrow, vertex_images, bdry_map=None):
        if bdry_map is None:
            bdry_map_key = None
        else:
            bdry_map_key = tuple(bdry_map)
        key = (arrow.Edge, arrow.Face, tuple(vertex_images), bdry_map_key)
        if key not in self.cache:
            self.cache[key] = TetrahedronEmbedding(arrow, vertex_images, bdry_map)
        return self.cache[key]