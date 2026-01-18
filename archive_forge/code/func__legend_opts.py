import copy
import math
import warnings
from types import FunctionType
import matplotlib.colors as mpl_colors
import numpy as np
import param
from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from packaging.version import Version
from ...core import (
from ...core.options import Keywords, abbreviated_exception
from ...element import Graph, Path
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_range_key, process_cmap
from .plot import MPLPlot, mpl_rc_context
from .util import EqHistNormalize, mpl_version, validate, wrap_formatter
@property
def _legend_opts(self):
    leg_spec = self.legend_specs[self.legend_position]
    if self.legend_cols:
        leg_spec['ncol'] = self.legend_cols
    legend_opts = self.legend_opts.copy()
    legend_opts.update(**dict(leg_spec, **self._fontsize('legend')))
    return legend_opts