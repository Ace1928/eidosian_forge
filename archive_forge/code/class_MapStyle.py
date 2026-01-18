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
class MapStyle(Style, Widget):
    """Map Style Widget

    Custom map style.

    Attributes
    ----------
    cursor: str, default 'grab'
        The cursor to use for the mouse when it's on the map. Should be a valid CSS
        cursor value.
    """
    _model_name = Unicode('LeafletMapStyleModel').tag(sync=True)
    _model_module = Unicode('jupyter-leaflet').tag(sync=True)
    _model_module_version = Unicode(EXTENSION_VERSION).tag(sync=True)
    cursor = Enum(values=allowed_cursor, default_value='grab').tag(sync=True)