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
class project_vectorfield(_project_operation):
    supported_types = [VectorField]

    def _calc_angles(self, ut, vt):
        return np.arctan2(vt, ut)

    def _process_element(self, element):
        if not len(element):
            return element.clone(crs=self.p.projection)
        xdim, ydim, adim, mdim = element.dimensions()[:4]
        xs, ys, ang, ms = (element.dimension_values(i) for i in range(4))
        coordinates = self.p.projection.transform_points(element.crs, xs, ys)
        mask = np.isfinite(coordinates[:, 0])
        new_data = {k: v[mask] for k, v in element.columns().items()}
        new_data[xdim.name] = coordinates[mask, 0]
        new_data[ydim.name] = coordinates[mask, 1]
        datatype = [element.interface.datatype] + element.datatype
        us = np.sin(ang) * -ms
        vs = np.cos(ang) * -ms
        ut, vt = self.p.projection.transform_vectors(element.crs, xs, ys, us, vs)
        with np.errstate(divide='ignore', invalid='ignore'):
            angle = self._calc_angles(ut, vt)
        mag = np.hypot(ut, vt)
        new_data[adim.name] = angle[mask]
        new_data[mdim.name] = mag[mask]
        if len(new_data[xdim.name]) == 0:
            self.param.warning(f'While projecting a {type(element).__name__} element from a {type(element.crs).__name__} coordinate reference system (crs) to a {type(self.p.projection).__name__} projection none of the projected paths were contained within the bounds specified by the projection. Ensure you have specified the correct coordinate system for your data.')
        return element.clone(tuple((new_data[d.name] for d in element.dimensions())), crs=self.p.projection, datatype=datatype)