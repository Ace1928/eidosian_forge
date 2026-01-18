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
class RasterLayer(Layer):
    """Abstract RasterLayer class.

    Attributes
    ----------
    opacity: float, default 1.
        Opacity of the layer between 0. (fully transparent) and 1. (fully opaque).
    visible: boolean, default True
        Whether the layer is visible or not.
    """
    _view_name = Unicode('LeafletRasterLayerView').tag(sync=True)
    _model_name = Unicode('LeafletRasterLayerModel').tag(sync=True)
    opacity = Float(1.0, min=0.0, max=1.0).tag(sync=True)
    visible = Bool(True).tag(sync=True)