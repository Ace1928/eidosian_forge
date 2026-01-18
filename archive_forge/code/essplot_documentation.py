import numpy as np
from bokeh.models import ColumnDataSource, Span
from bokeh.models.annotations import Legend, Title
from scipy.stats import rankdata
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
from ...plot_utils import _scale_fig_size
from bokeh.models.glyphs import Scatter
Bokeh essplot.