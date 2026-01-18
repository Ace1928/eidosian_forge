import logging
import sys
import param
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from holoviews.core.data import MultiInterface
from holoviews.core.util import cartesian_product, get_param_values
from holoviews.operation import Operation
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from ..data import GeoPandasInterface
from ..element import (Image, Shape, Polygons, Path, Points, Contours,
from ..util import (
class project_shape(_project_operation):
    """
    Projects Shape Element from the source coordinate reference system
    to the supplied projection.
    """
    supported_types = [Shape]

    def _process_element(self, element):
        if not len(element):
            return element.clone(crs=self.p.projection)
        geom = element.geom()
        if isinstance(geom, (MultiPolygon, Polygon)):
            obj = Polygons([geom])
        else:
            obj = Path([geom])
        geom = project_path(obj, projection=self.p.projection).geom()
        return element.clone(geom, crs=self.p.projection)