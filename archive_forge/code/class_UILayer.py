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
class UILayer(Layer):
    """Abstract UILayer class."""
    _view_name = Unicode('LeafletUILayerView').tag(sync=True)
    _model_name = Unicode('LeafletUILayerModel').tag(sync=True)