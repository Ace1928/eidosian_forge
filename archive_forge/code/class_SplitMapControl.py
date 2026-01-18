import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
class SplitMapControl(Control):
    """SplitMapControl class, with Control as parent class.

    A control which allows comparing layers by splitting the map in two.

    Attributes
    ----------
    left_layer: Layer or list of Layers
        The left layer(s) for comparison.
    right_layer: Layer or list of Layers
        The right layer(s) for comparison.
    """
    _view_name = Unicode('LeafletSplitMapControlView').tag(sync=True)
    _model_name = Unicode('LeafletSplitMapControlModel').tag(sync=True)
    left_layer = Union((Instance(Layer), List(Instance(Layer)))).tag(sync=True, **widget_serialization)
    right_layer = Union((Instance(Layer), List(Instance(Layer)))).tag(sync=True, **widget_serialization)

    @default('left_layer')
    def _default_left_layer(self):
        return TileLayer()

    @default('right_layer')
    def _default_right_layer(self):
        return TileLayer()

    def __init__(self, **kwargs):
        super(SplitMapControl, self).__init__(**kwargs)
        self.on_msg(self._handle_leaflet_event)

    def _handle_leaflet_event(self, _, content, buffers):
        if content.get('event', '') == 'dividermove':
            event = content.get('event')
            self.x = event.x