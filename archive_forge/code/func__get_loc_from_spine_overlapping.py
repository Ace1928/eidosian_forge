import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _get_loc_from_spine_overlapping(self, spines_specs, xylabel, label_path):
    """Try to get the location from side spines and label path

        Returns None if it does not apply

        For instance, for each side, if any of label_path x coordinates
        are beyond this side, the distance to this side is computed.
        If several sides are matching (max 2), then the one with a greater
        distance is kept.

        This helps finding the side of labels for non-rectangular projection
        with a rectangular map boundary.

        """
    side_max = dist_max = None
    for side, specs in spines_specs.items():
        if specs['coord_type'] == xylabel:
            continue
        label_coords = label_path.vertices[:-1, specs['index']]
        spine_coord = specs['opval'](specs['coords'])
        if not specs['opcmp'](label_coords, spine_coord).any():
            continue
        if specs['opcmp'] is operator.ge:
            dist = label_coords.min() - spine_coord
        else:
            dist = spine_coord - label_coords.max()
        if side_max is None or dist > dist_max:
            side_max = side
            dist_max = dist
    if side_max is None:
        return 'geo'
    return side_max