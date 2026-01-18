from __future__ import annotations
import sys
from collections import defaultdict
from functools import partial
from typing import (
import param
from bokeh.models import Range1d, Spacer as _BkSpacer
from bokeh.themes.theme import Theme
from packaging.version import Version
from param.parameterized import register_reference_transform
from param.reactive import bind
from ..io import state, unlocked
from ..layout import (
from ..viewable import Layoutable, Viewable
from ..widgets import Player
from .base import PaneBase, RerenderError, panel
from .plot import Bokeh, Matplotlib
from .plotly import Plotly
def _update_plot(self, plot, pane):
    from holoviews.core.util import cross_index, wrap_tuple_streams
    widgets = self.widget_box.objects
    if not widgets:
        return
    elif self.widget_type == 'scrubber':
        key = cross_index([v for v in self._values.values()], widgets[0].value)
    else:
        key = tuple((w.value for w in widgets))
        if plot.dynamic:
            widget_dims = [w.name for w in widgets]
            dim_labels = [kdim.pprint_label for kdim in plot.dimensions]
            key = [key[widget_dims.index(kdim)] if kdim in widget_dims else None for kdim in dim_labels]
            key = wrap_tuple_streams(tuple(key), plot.dimensions, plot.streams)
    if plot.backend == 'bokeh':
        if plot.comm or state._unblocked(plot.document):
            with unlocked():
                plot.update(key)
            if plot.comm and 'embedded' not in plot.root.tags:
                plot.push()
        elif plot.document.session_context:
            plot.document.add_next_tick_callback(partial(plot.update, key))
        else:
            plot.update(key)
    else:
        plot.update(key)
        if hasattr(plot.renderer, 'get_plot_state'):
            pane.object = plot.renderer.get_plot_state(plot)
        else:
            pane.object = plot.state