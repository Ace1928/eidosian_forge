import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
def _get_bbox_and_child_offsets(self, renderer):
    pad = self.pad * renderer.points_to_pixels(1.0)
    return (self._children[0].get_bbox(renderer).padded(pad), [(0, 0)])