import warnings
import bokeh.plotting as bkp
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Title
from bokeh.models.glyphs import Scatter
from ....rcparams import _validate_bokeh_marker, rcParams
from ...plot_utils import _scale_fig_size, color_from_dim, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def _plot_atomic_elpd(ax_, xdata, ydata, model1, model2, threshold, coord_labels, xlabels, xlabels_shown, ylabels_shown, plot_kwargs):
    marker = _validate_bokeh_marker(plot_kwargs.get('marker'))
    sizes = np.ones(len(xdata)) * plot_kwargs.get('s')
    glyph = Scatter(x='xdata', y='ydata', size='sizes', line_color=plot_kwargs.get('color', 'black'), marker=marker)
    source = ColumnDataSource(dict(xdata=xdata, ydata=ydata, sizes=sizes))
    ax_.add_glyph(source, glyph)
    if threshold is not None:
        diff_abs = np.abs(ydata - ydata.mean())
        bool_ary = diff_abs > threshold * ydata.std()
        if coord_labels is None:
            coord_labels = xdata.astype(str)
        outliers = np.nonzero(bool_ary)[0]
        for outlier in outliers:
            label = coord_labels[outlier]
            ax_.text(x=[outlier], y=[ydata[outlier]], text=label, text_color='black')
    if ylabels_shown:
        ax_.yaxis.axis_label = 'ELPD difference'
    else:
        ax_.yaxis.minor_tick_line_color = None
        ax_.yaxis.major_label_text_font_size = '0pt'
    if xlabels_shown:
        if xlabels:
            ax_.xaxis.ticker = np.arange(0, len(coord_labels))
            ax_.xaxis.major_label_overrides = {str(key): str(value) for key, value in zip(np.arange(0, len(coord_labels)), list(coord_labels))}
    else:
        ax_.xaxis.minor_tick_line_color = None
        ax_.xaxis.major_label_text_font_size = '0pt'
    title = Title()
    title.text = f'{model1} - {model2}'
    ax_.title = title