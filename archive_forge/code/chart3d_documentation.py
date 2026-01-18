import numpy as np
import param
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from packaging.version import Version
from ...core import Dimension
from ...core.options import abbreviated_exception
from ...util.transform import dim as dim_expr
from ..util import map_colors
from .chart import PointPlot
from .element import ColorbarPlot
from .path import PathPlot
from .util import mpl_version

        Extends the ElementPlot _finalize_axis method to set appropriate
        labels, and axes options for 3D Plots.
        