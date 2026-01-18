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
class project_geom(_project_operation):
    supported_types = [Rectangles, Segments]

    def _process_element(self, element):
        x0d, y0d, x1d, y1d = element.kdims
        x0, y0, x1, y1 = (element.dimension_values(i) for i in range(4))
        p1 = self.p.projection.transform_points(element.crs, x0, y0)
        p2 = self.p.projection.transform_points(element.crs, x1, y1)
        mask = np.isfinite(p1[:, 0]) & np.isfinite(p2[:, 0])
        new_data = {k: v[mask] for k, v in element.columns(element.vdims).items()}
        new_data[x0d.name] = p1[mask, 0]
        new_data[y0d.name] = p1[mask, 1]
        new_data[x1d.name] = p2[mask, 0]
        new_data[y1d.name] = p2[mask, 1]
        if len(new_data[x0d.name]) == 0:
            element_name = type(element).__name__
            crs_name = type(element.crs).__name__
            proj_name = type(self.p.projection).__name__
            self.param.warning(f'While projecting a {element_name} element from a {crs_name} coordinate reference system (crs) to a {proj_name} projection none of the projected paths were contained within the bounds specified by the projection. Ensure you have specified the correct coordinate system for your data.')
        return element.clone(tuple((new_data[d.name] for d in element.dimensions())), crs=self.p.projection)