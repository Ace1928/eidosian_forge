import param
import numpy as np
from bokeh.models import MercatorTileSource
from cartopy import crs as ccrs
from cartopy.feature import Feature as cFeature
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.shapereader import Reader
from holoviews.core import Element2D, Dimension, Dataset as HvDataset, NdOverlay, Overlay
from holoviews.core import util
from holoviews.element import (
from holoviews.element.selection import Selection2DExpr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import (
from shapely.ops import unary_union
from ..util import (
class WMTS(_GeoFeature):
    """
    The WMTS Element represents a Web Map Tile Service specified as URL
    containing different template variables or xyzservices.TileProvider.

    These variables correspond to three different formats for specifying the spatial
    location and zoom level of the requested tiles:

    * Web mapping tiles sources containing {x}, {y}, and {z} variables
    * Bounding box tile sources containing {XMIN}, {XMAX}, {YMIN}, {YMAX} variables
    * Quadkey tile sources containing a {Q} variable

    Tiles are defined in a pseudo-Mercator projection (EPSG:3857)
    defined as eastings and northings. Any data overlaid on a tile
    source therefore has to be defined in those coordinates or be
    projected.
    """
    crs = param.ClassSelector(default=ccrs.GOOGLE_MERCATOR, class_=ccrs.CRS, doc='\n        Cartopy coordinate-reference-system specifying the\n        coordinate system of the data. Inferred automatically\n        when _Element wraps cartopy Feature object.')
    group = param.String(default='WMTS')
    layer = param.String(doc='The layer on the tile service')

    def __init__(self, data, kdims=None, vdims=None, **params):
        if MercatorTileSource and isinstance(data, MercatorTileSource) or (GoogleTiles and isinstance(data, GoogleTiles)):
            data = data.url
        elif WebMapTileService and isinstance(data, WebMapTileService):
            pass
        elif data is not None and (not isinstance(data, (str, dict))):
            raise TypeError(f'{type(self).__name__} data should be a tile service URL or xyzservices.TileProvider not a {type(data).__name__} type.')
        super().__init__(data, kdims=kdims, vdims=vdims, **params)

    def __call__(self, *args, **kwargs):
        return self.opts(*args, **kwargs)