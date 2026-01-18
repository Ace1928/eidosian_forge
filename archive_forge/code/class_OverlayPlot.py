import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
class OverlayPlot(GenericOverlayPlot, ElementPlot):
    _propagate_options = ['width', 'height', 'xaxis', 'yaxis', 'labelled', 'bgcolor', 'invert_axes', 'show_frame', 'show_grid', 'logx', 'logy', 'xticks', 'toolbar', 'yticks', 'xrotation', 'yrotation', 'responsive', 'invert_xaxis', 'invert_yaxis', 'sizing_mode', 'title', 'title_format', 'padding', 'xlabel', 'ylabel', 'zlabel', 'xlim', 'ylim', 'zlim']

    def initialize_plot(self, ranges=None, is_geo=False):
        """
        Initializes a new plot object with the last available frame.
        """
        return self.generate_plot(next(iter(self.hmap.data.keys())), ranges, is_geo=is_geo)

    def generate_plot(self, key, ranges, element=None, is_geo=False):
        if element is None:
            element = self._get_frame(key)
        items = [] if element is None else list(element.data.items())
        plot_opts = self.lookup_options(element, 'plot').options
        inherited = self._traverse_options(element, 'plot', self._propagate_options, defaults=False)
        plot_opts.update(**{k: v[0] for k, v in inherited.items() if k not in plot_opts})
        self.param.update(**plot_opts)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        figure = None
        for _, el in items:
            if isinstance(el, Tiles):
                is_geo = True
                break
        for okey, subplot in self.subplots.items():
            if element is not None and subplot.drawn:
                idx, spec, exact = self._match_subplot(okey, subplot, items, element)
                if idx is not None:
                    _, el = items.pop(idx)
                else:
                    el = None
            else:
                el = None
            subplot.param.update(**plot_opts)
            fig = subplot.generate_plot(key, ranges, el, is_geo=is_geo)
            if figure is None:
                figure = fig
            else:
                merge_figure(figure, fig)
        layout = self.init_layout(key, element, ranges, is_geo=is_geo)
        merge_layout(figure['layout'], layout)
        self.drawn = True
        self.handles['fig'] = figure
        return figure

    def update_frame(self, key, ranges=None, element=None, is_geo=False):
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        self.prev_frame = self.current_frame
        if not reused and element is None:
            element = self._get_frame(key)
        elif element is not None:
            self.current_frame = element
            self.current_key = key
        items = [] if element is None else list(element.data.items())
        for _, el in items:
            if isinstance(el, Tiles):
                is_geo = True
        for k, subplot in self.subplots.items():
            if not (isinstance(self.hmap, DynamicMap) and element is not None):
                continue
            idx, _, _ = self._match_subplot(k, subplot, items, element)
            if idx is not None:
                items.pop(idx)
        if isinstance(self.hmap, DynamicMap) and items:
            self._create_dynamic_subplots(key, items, ranges)
        self.generate_plot(key, ranges, element, is_geo=is_geo)