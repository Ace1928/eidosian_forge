import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
def _get_sindex_class():
    """Dynamically chooses a spatial indexing backend.

    Required to comply with _compat.USE_PYGEOS.
    The selection order goes PyGEOS > RTree > Error.
    """
    if compat.USE_SHAPELY_20 or compat.USE_PYGEOS:
        return PyGEOSSTRTreeIndex
    if compat.HAS_RTREE:
        return RTreeIndex
    raise ImportError('Spatial indexes require either `rtree` or `pygeos`. See installation instructions at https://geopandas.org/install.html')