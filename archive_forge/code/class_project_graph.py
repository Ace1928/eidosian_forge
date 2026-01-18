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
class project_graph(_project_operation):
    supported_types = [Graph]

    def _process_element(self, element):
        proj = self.p.projection
        nodes = project_points(element.nodes, projection=proj)
        data = (element.data, nodes)
        if element._edgepaths:
            data = data + (project_path(element.edgepaths, projection=proj),)
        return element.clone(data, crs=proj)