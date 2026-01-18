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
class CircleMarker(Path):
    """CircleMarker class, with Path as parent class.

    CircleMarker layer.

    Attributes
    ----------
    location: list, default [0, 0]
        Location of the marker (lat, long).
    radius: int, default 10
        Radius of the circle marker in pixels.
    """
    _view_name = Unicode('LeafletCircleMarkerView').tag(sync=True)
    _model_name = Unicode('LeafletCircleMarkerModel').tag(sync=True)
    location = List(def_loc).tag(sync=True)
    radius = Int(10, help='radius of circle in pixels').tag(sync=True, o=True)