from contextlib import contextmanager
from itertools import chain
import matplotlib as mpl
import numpy as np
import param
from matplotlib import (
from matplotlib.font_manager import font_scalings
from mpl_toolkits.mplot3d import Axes3D  # noqa (For 3D plots)
from ...core import (
from ...core.options import SkipRendering, Store
from ...core.util import int_to_alpha, int_to_roman, wrap_tuple_streams
from ..plot import (
from ..util import attach_streams, collate, displayable
from .util import compute_ratios, fix_aspect, get_old_rcparams
class AdjoinedPlot(DimensionedPlot):
    aspect = param.Parameter(default='auto', doc='\n        Aspect ratios on SideHistogramPlot should be determined by the\n        AdjointLayoutPlot.')
    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc='\n        Make plot background invisible.')
    border_size = param.Number(default=0.25, doc='\n        The size of the border expressed as a fraction of the main plot.')
    show_title = param.Boolean(default=False, doc='\n        Titles should be disabled on all SidePlots to avoid clutter.')
    subplot_size = param.Number(default=0.25, doc='\n        The size subplots as expressed as a fraction of the main plot.')
    show_xlabel = param.Boolean(default=False, doc='\n        Whether to show the x-label of the plot. Disabled by default\n        because plots are often too cramped to fit the title correctly.')