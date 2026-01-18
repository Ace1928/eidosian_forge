from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def _3d_extend_contour(self, cset, stride=5):
    """
        Extend a contour in 3D by creating
        """
    dz = (cset.levels[1] - cset.levels[0]) / 2
    polyverts = []
    colors = []
    for idx, level in enumerate(cset.levels):
        path = cset.get_paths()[idx]
        subpaths = [*path._iter_connected_components()]
        color = cset.get_edgecolor()[idx]
        top = art3d._paths_to_3d_segments(subpaths, level - dz)
        bot = art3d._paths_to_3d_segments(subpaths, level + dz)
        if not len(top[0]):
            continue
        nsteps = max(round(len(top[0]) / stride), 2)
        stepsize = (len(top[0]) - 1) / (nsteps - 1)
        polyverts.extend([(top[0][round(i * stepsize)], top[0][round((i + 1) * stepsize)], bot[0][round((i + 1) * stepsize)], bot[0][round(i * stepsize)]) for i in range(round(nsteps) - 1)])
        colors.extend([color] * (round(nsteps) - 1))
    self.add_collection3d(art3d.Poly3DCollection(np.array(polyverts), facecolors=colors, edgecolors=colors, shade=True))
    cset.remove()