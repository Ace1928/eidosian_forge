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
class project(Operation):
    """
    Projects GeoViews Element types to the specified projection.
    """
    projection = param.ClassSelector(default=ccrs.GOOGLE_MERCATOR, class_=ccrs.Projection, instantiate=False, doc='\n        Projection the image type is projected to.')
    _operations = [project_path, project_image, project_shape, project_graph, project_quadmesh, project_points, project_vectorfield, project_windbarbs, project_geom]

    def _process(self, element, key=None):
        for op in self._operations:
            element = element.map(op.instance(projection=self.p.projection), op.supported_types)
        return element