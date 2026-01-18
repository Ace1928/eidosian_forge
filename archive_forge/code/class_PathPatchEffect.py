from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
class PathPatchEffect(AbstractPathEffect):
    """
    Draws a `.PathPatch` instance whose Path comes from the original
    PathEffect artist.
    """

    def __init__(self, offset=(0, 0), **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        **kwargs
            All keyword arguments are passed through to the
            :class:`~matplotlib.patches.PathPatch` constructor. The
            properties which cannot be overridden are "path", "clip_box"
            "transform" and "clip_path".
        """
        super().__init__(offset=offset)
        self.patch = mpatches.PathPatch([], **kwargs)

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        self.patch._path = tpath
        self.patch.set_transform(affine + self._offset_transform(renderer))
        self.patch.set_clip_box(gc.get_clip_rectangle())
        clip_path = gc.get_clip_path()
        if clip_path and self.patch.get_clip_path() is None:
            self.patch.set_clip_path(*clip_path)
        self.patch.draw(renderer)