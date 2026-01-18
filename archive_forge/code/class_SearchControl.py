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
class SearchControl(Control):
    """ SearchControl class, with Control as parent class.

    Attributes
    ----------

    url: string, default ""
        The url used for the search queries.
    layer:	default None
        The LayerGroup used for search queries.
    zoom: int, default None
        The zoom level after moving to searched location, by default zoom level will not change.
    marker:	default Marker()
        The marker used by the control.
    found_style: default {‘fillColor’: ‘#3f0’, ‘color’: ‘#0f0’}
        Style for searched feature when searching in LayerGroup.
    """
    _view_name = Unicode('LeafletSearchControlView').tag(sync=True)
    _model_name = Unicode('LeafletSearchControlModel').tag(sync=True)
    url = Unicode().tag(sync=True, o=True)
    zoom = Int(default_value=None, allow_none=True).tag(sync=True, o=True)
    property_name = Unicode('display_name').tag(sync=True, o=True)
    property_loc = List(['lat', 'lon']).tag(sync=True, o=True)
    jsonp_param = Unicode('json_callback').tag(sync=True, o=True)
    auto_type = Bool(False).tag(sync=True, o=True)
    auto_collapse = Bool(False).tag(sync=True, o=True)
    animate_location = Bool(False).tag(sync=True, o=True)
    found_style = Dict(default_value={'fillColor': '#3f0', 'color': '#0f0'}).tag(sync=True, o=True)
    marker = Instance(Marker, allow_none=True, default_value=None).tag(sync=True, **widget_serialization)
    layer = Instance(LayerGroup, allow_none=True, default_value=None).tag(sync=True, **widget_serialization)
    _location_found_callbacks = Instance(CallbackDispatcher, ())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_msg(self._handle_leaflet_event)

    def _handle_leaflet_event(self, _, content, buffers):
        if content.get('event', '') == 'locationfound':
            self._location_found_callbacks(**content)

    def on_feature_found(self, callback, remove=False):
        """Add a found feature event listener for searching in GeoJSON layer.

        Parameters
        ----------
        callback : callable
            Callback function that will be called on found event when searching in GeoJSON layer.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
        self._location_found_callbacks.register_callback(callback, remove=remove)

    def on_location_found(self, callback, remove=False):
        """Add a found location event listener. The callback will be called when a search result has been found.

        Parameters
        ----------
        callback : callable
            Callback function that will be called on location found event.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
        self._location_found_callbacks.register_callback(callback, remove=remove)