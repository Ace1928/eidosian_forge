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
def add_collection3d(self, col, zs=0, zdir='z'):
    """
        Add a 3D collection object to the plot.

        2D collection types are converted to a 3D version by
        modifying the object and adding z coordinate information.

        Supported are:

        - PolyCollection
        - LineCollection
        - PatchCollection
        """
    zvals = np.atleast_1d(zs)
    zsortval = np.min(zvals) if zvals.size else 0
    if type(col) is mcoll.PolyCollection:
        art3d.poly_collection_2d_to_3d(col, zs=zs, zdir=zdir)
        col.set_sort_zpos(zsortval)
    elif type(col) is mcoll.LineCollection:
        art3d.line_collection_2d_to_3d(col, zs=zs, zdir=zdir)
        col.set_sort_zpos(zsortval)
    elif type(col) is mcoll.PatchCollection:
        art3d.patch_collection_2d_to_3d(col, zs=zs, zdir=zdir)
        col.set_sort_zpos(zsortval)
    collection = super().add_collection(col)
    return collection