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
class WKTLayer(GeoJSON):
    """WKTLayer class.

    Layer created from a local WKT file or WKT string input.

    Attributes
    ----------
    path: string, default ""
      file path of local WKT file.
    wkt_string: string, default ""
      WKT string.
    """
    path = Unicode('')
    wkt_string = Unicode('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self._get_data()

    @observe('path', 'wkt_string', 'style', 'style_callback')
    def _update_data(self, change):
        self.data = self._get_data()

    def _get_data(self):
        try:
            from shapely import geometry, wkt
        except ImportError:
            raise RuntimeError('The WKTLayer needs shapely to be installed, please run `pip install shapely`')
        if self.path:
            with open(self.path) as f:
                parsed_wkt = wkt.load(f)
        elif self.wkt_string:
            parsed_wkt = wkt.loads(self.wkt_string)
        else:
            raise ValueError('Please provide either WKT file path or WKT string')
        geo = geometry.mapping(parsed_wkt)
        if geo['type'] == 'GeometryCollection':
            features = [{'geometry': g, 'properties': {}, 'type': 'Feature'} for g in geo['geometries']]
            feature_collection = {'type': 'FeatureCollection', 'features': features}
            return feature_collection
        else:
            feature = {'geometry': geo, 'properties': {}, 'type': 'Feature'}
            return feature