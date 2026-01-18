from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
def _offset_transform(self, renderer):
    """Apply the offset to the given transform."""
    return mtransforms.Affine2D().translate(*map(renderer.points_to_pixels, self._offset))