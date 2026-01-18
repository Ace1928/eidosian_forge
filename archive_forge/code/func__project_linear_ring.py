from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def _project_linear_ring(self, linear_ring, src_crs):
    """
        Project the given LinearRing from the src_crs into this CRS and
        returns a list of LinearRings and a single MultiLineString.

        """
    debug = False
    multi_line_string = cartopy.trace.project_linear(linear_ring, src_crs, self)
    threshold = max(np.abs(self.x_limits + self.y_limits)) * 1e-05
    if len(multi_line_string.geoms) > 1:
        line_strings = list(multi_line_string.geoms)
        any_modified = False
        i = 0
        if debug:
            first_coord = np.array([ls.coords[0] for ls in line_strings])
            last_coord = np.array([ls.coords[-1] for ls in line_strings])
            print('Distance matrix:')
            np.set_printoptions(precision=2)
            x = first_coord[:, np.newaxis, :]
            y = last_coord[np.newaxis, :, :]
            print(np.abs(x - y).max(axis=-1))
        while i < len(line_strings):
            modified = False
            j = 0
            while j < len(line_strings):
                if i != j and np.allclose(line_strings[i].coords[0], line_strings[j].coords[-1], atol=threshold):
                    if debug:
                        print(f'Joining together {i} and {j}.')
                    last_coords = list(line_strings[j].coords)
                    first_coords = list(line_strings[i].coords)[1:]
                    combo = sgeom.LineString(last_coords + first_coords)
                    if j < i:
                        i, j = (j, i)
                    del line_strings[j], line_strings[i]
                    line_strings.append(combo)
                    modified = True
                    any_modified = True
                    break
                else:
                    j += 1
            if not modified:
                i += 1
        if any_modified:
            multi_line_string = sgeom.MultiLineString(line_strings)
    rings = []
    line_strings = []
    for line in multi_line_string.geoms:
        if len(line.coords) > 3 and np.allclose(line.coords[0], line.coords[-1], atol=threshold):
            result_geometry = sgeom.LinearRing(line.coords[:-1])
            rings.append(result_geometry)
        else:
            line_strings.append(line)
    if rings:
        multi_line_string = sgeom.MultiLineString(line_strings)
    return (rings, multi_line_string)