import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
class TriMesh(Collection):
    """
    Class for the efficient drawing of a triangular mesh using Gouraud shading.

    A triangular mesh is a `~matplotlib.tri.Triangulation` object.
    """

    def __init__(self, triangulation, **kwargs):
        super().__init__(**kwargs)
        self._triangulation = triangulation
        self._shading = 'gouraud'
        self._bbox = transforms.Bbox.unit()
        xy = np.hstack((triangulation.x.reshape(-1, 1), triangulation.y.reshape(-1, 1)))
        self._bbox.update_from_data_xy(xy)

    def get_paths(self):
        if self._paths is None:
            self.set_paths()
        return self._paths

    def set_paths(self):
        self._paths = self.convert_mesh_to_paths(self._triangulation)

    @staticmethod
    def convert_mesh_to_paths(tri):
        """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support meshes.
        """
        triangles = tri.get_masked_triangles()
        verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)
        return [mpath.Path(x) for x in verts]

    @artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__, gid=self.get_gid())
        transform = self.get_transform()
        tri = self._triangulation
        triangles = tri.get_masked_triangles()
        verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)
        self.update_scalarmappable()
        colors = self._facecolors[triangles]
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])
        renderer.draw_gouraud_triangles(gc, verts, colors, transform.frozen())
        gc.restore()
        renderer.close_group(self.__class__.__name__)