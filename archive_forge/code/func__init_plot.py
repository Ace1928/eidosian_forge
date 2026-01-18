import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def _init_plot(self, key, element, plots, ranges=None):
    """
        Initializes Bokeh figure to draw Element into and sets basic
        figure and axis attributes including axes types, labels,
        titles and plot height and width.
        """
    subplots = list(self.subplots.values()) if self.subplots else []
    axis_specs = {'x': {}, 'y': {}}
    axis_specs['x']['x'] = self._axis_props(plots, subplots, element, ranges, pos=0) + (self.xaxis, {})
    if self.multi_y:
        if not bokeh32:
            self.param.warning('Independent axis zooming for multi_y=True only supported for Bokeh >=3.2')
        yaxes, extra_axis_specs = self._create_extra_axes(plots, subplots, element, ranges)
        axis_specs['y'].update(extra_axis_specs)
    else:
        range_tags_extras = {'invert_yaxis': self.invert_yaxis}
        if self.autorange == 'y':
            range_tags_extras['autorange'] = True
            lowerlim, upperlim = self.ylim
            if not (lowerlim is None or np.isnan(lowerlim)):
                range_tags_extras['y-lowerlim'] = lowerlim
            if not (upperlim is None or np.isnan(upperlim)):
                range_tags_extras['y-upperlim'] = upperlim
        else:
            range_tags_extras['autorange'] = False
        axis_specs['y']['y'] = self._axis_props(plots, subplots, element, ranges, pos=1, range_tags_extras=range_tags_extras) + (self.yaxis, {})
    if self._subcoord_overlaid:
        _, extra_axis_specs = self._create_extra_axes(plots, subplots, element, ranges)
        axis_specs['y'].update(extra_axis_specs)
    properties, axis_props = ({}, {'x': {}, 'y': {}})
    for axis, axis_spec in axis_specs.items():
        for axis_dim, (axis_type, axis_label, axis_range, axis_position, fontsize) in axis_spec.items():
            scale = get_scale(axis_range, axis_type)
            if f'{axis}_range' in properties:
                properties[f'extra_{axis}_ranges'] = extra_ranges = properties.get(f'extra_{axis}_ranges', {})
                extra_ranges[axis_dim] = axis_range
                if not self.subcoordinate_y:
                    properties[f'extra_{axis}_scales'] = extra_scales = properties.get(f'extra_{axis}_scales', {})
                    extra_scales[axis_dim] = scale
            else:
                properties[f'{axis}_range'] = axis_range
                properties[f'{axis}_scale'] = scale
                properties[f'{axis}_axis_type'] = axis_type
                if axis_label and axis in self.labelled:
                    properties[f'{axis}_axis_label'] = axis_label
                locs = {'left': 'left', 'right': 'right'} if axis == 'y' else {'bottom': 'below', 'top': 'above'}
                if axis_position is None:
                    axis_props[axis]['visible'] = False
                axis_props[axis].update(fontsize)
                for loc, pos in locs.items():
                    if axis_position and loc in axis_position:
                        properties[f'{axis}_axis_location'] = pos
    if not self.show_frame:
        properties['outline_line_alpha'] = 0
    if self.show_title and self.adjoined is None:
        title = self._format_title(key, separator=' ')
    else:
        title = ''
    if self.toolbar != 'disable':
        tools = self._init_tools(element)
        properties['tools'] = tools
        properties['toolbar_location'] = self.toolbar
    else:
        properties['tools'] = []
        properties['toolbar_location'] = None
    if self.renderer.webgl:
        properties['output_backend'] = 'webgl'
    properties.update(**self._plot_properties(key, element))
    figure = bokeh.plotting.figure
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig = figure(title=title, **properties)
    fig.xaxis[0].update(**axis_props['x'])
    fig.yaxis[0].update(**axis_props['y'])
    if self._subcoord_overlaid:
        return fig
    multi_ax = 'x' if self.invert_axes else 'y'
    for axis_dim, range_obj in properties.get(f'extra_{multi_ax}_ranges', {}).items():
        axis_type, axis_label, _, axis_position, fontsize = axis_specs[multi_ax][axis_dim]
        ax_cls, ax_kwargs = get_axis_class(axis_type, range_obj, dim=1)
        ax_kwargs[f'{multi_ax}_range_name'] = axis_dim
        ax_kwargs.update(fontsize)
        if axis_position is None:
            ax_kwargs['visible'] = False
            axis_position = 'above' if multi_ax == 'x' else 'right'
        if multi_ax in self.labelled:
            ax_kwargs['axis_label'] = axis_label
        ax = ax_cls(**ax_kwargs)
        fig.add_layout(ax, axis_position)
    return fig