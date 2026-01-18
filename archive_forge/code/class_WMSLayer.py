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
class WMSLayer(TileLayer):
    """WMSLayer class, with TileLayer as a parent class.

    Attributes
    ----------
    layers: string, default ""
        Comma-separated list of WMS layers to show.
    styles: string, default ""
        Comma-separated list of WMS styles
    format: string, default "image/jpeg"
        WMS image format (use `'image/png'` for layers with transparency).
    transparent: boolean, default False
        If true, the WMS service will return images with transparency.
    crs: dict, default ipyleaflet.projections.EPSG3857
        Projection used for this WMS service.
    """
    _view_name = Unicode('LeafletWMSLayerView').tag(sync=True)
    _model_name = Unicode('LeafletWMSLayerModel').tag(sync=True)
    layers = Unicode().tag(sync=True, o=True)
    styles = Unicode().tag(sync=True, o=True)
    format = Unicode('image/jpeg').tag(sync=True, o=True)
    transparent = Bool(False).tag(sync=True, o=True)
    crs = Dict(default_value=projections.EPSG3857).tag(sync=True)
    uppercase = Bool(False).tag(sync=True, o=True)