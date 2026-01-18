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
class project_points(_project_operation):
    supported_types = [Points, Nodes, HexTiles, Labels]

    def _process_element(self, element):
        if not len(element):
            return element.clone(crs=self.p.projection)
        xdim, ydim = element.dimensions()[:2]
        xs, ys = (element.dimension_values(i) for i in range(2))
        coordinates = self.p.projection.transform_points(element.crs, np.asarray(xs), np.asarray(ys))
        mask = np.isfinite(coordinates[:, 0])
        dims = [d for d in element.dimensions() if d not in (xdim, ydim)]
        new_data = {k: v[mask] for k, v in element.columns(dims).items()}
        new_data[xdim.name] = coordinates[mask, 0]
        new_data[ydim.name] = coordinates[mask, 1]
        if len(new_data[xdim.name]) == 0:
            element_name = type(element).__name__
            crs_name = type(element.crs).__name__
            proj_name = type(self.p.projection).__name__
            self.param.warning(f'While projecting a {element_name} element from a {crs_name} coordinate reference system (crs) to a {proj_name} projection none of the projected paths were contained within the bounds specified by the projection. Ensure you have specified the correct coordinate system for your data.')
        return element.clone(tuple((new_data[d.name] for d in element.dimensions())), crs=self.p.projection)