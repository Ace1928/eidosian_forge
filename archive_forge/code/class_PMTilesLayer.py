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
class PMTilesLayer(Layer):
    """PMTilesLayer class, with Layer as parent class.

    PMTiles layer.


    Attributes
    ----------
    url: string, default ""
        Url to the PMTiles archive.
    attribution: string, default ""
        PMTiles archive attribution.
    style: dict, default {}
        CSS Styles to apply to the vector data.
    """
    _view_name = Unicode('LeafletPMTilesLayerView').tag(sync=True)
    _model_name = Unicode('LeafletPMTilesLayerModel').tag(sync=True)
    url = Unicode().tag(sync=True, o=True)
    attribution = Unicode().tag(sync=True, o=True)
    style = Dict().tag(sync=True, o=True)

    def add_inspector(self):
        """Add an inspector to the layer.
        """
        self.send({'msg': 'add_inspector'})