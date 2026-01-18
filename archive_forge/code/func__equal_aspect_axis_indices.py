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
def _equal_aspect_axis_indices(self, aspect):
    """
        Get the indices for which of the x, y, z axes are constrained to have
        equal aspect ratios.

        Parameters
        ----------
        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
            See descriptions in docstring for `.set_aspect()`.
        """
    ax_indices = []
    if aspect == 'equal':
        ax_indices = [0, 1, 2]
    elif aspect == 'equalxy':
        ax_indices = [0, 1]
    elif aspect == 'equalxz':
        ax_indices = [0, 2]
    elif aspect == 'equalyz':
        ax_indices = [1, 2]
    return ax_indices