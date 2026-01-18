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
class VectorTileLayer(Layer):
    """VectorTileLayer class, with Layer as parent class.

    Vector tile layer.


    Attributes
    ----------
    url: string, default ""
        Url to the vector tile service.
    attribution: string, default ""
        Vector tile service attribution.
    vector_tile_layer_styles: dict, default {}
        CSS Styles to apply to the vector data.
    """
    _view_name = Unicode('LeafletVectorTileLayerView').tag(sync=True)
    _model_name = Unicode('LeafletVectorTileLayerModel').tag(sync=True)
    url = Unicode().tag(sync=True, o=True)
    attribution = Unicode().tag(sync=True, o=True)
    vector_tile_layer_styles = Dict().tag(sync=True, o=True)

    def redraw(self):
        """Force redrawing the tiles.

        This is especially useful when you are sure the server updated the tiles and you
        need to refresh the layer.
        """
        self.send({'msg': 'redraw'})