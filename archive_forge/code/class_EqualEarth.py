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
class EqualEarth(_WarpedRectangularProjection):
    """
    An Equal Earth projection.

    This projection is pseudocylindrical, and equal area. Parallels are
    unequally-spaced straight lines, while meridians are equally-spaced arcs.

    It is intended for world maps.

    Note
    ----
    To use this projection, you must be using Proj 5.2.0 or newer.

    References
    ----------
    Bojan Šavrič, Tom Patterson & Bernhard Jenny (2018)
    The Equal Earth map projection,
    International Journal of Geographical Information Science,
    DOI: 10.1080/13658816.2018.1504949

    """

    def __init__(self, central_longitude=0, false_easting=None, false_northing=None, globe=None):
        """
        Parameters
        ----------
        central_longitude: float, optional
            The central longitude. Defaults to 0.
        false_easting: float, optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: float, optional
            Y offset from planar origin in metres. Defaults to 0.
        globe: :class:`cartopy.crs.Globe`, optional
            If omitted, a default globe is created.

        """
        proj_params = [('proj', 'eqearth'), ('lon_0', central_longitude)]
        super().__init__(proj_params, central_longitude, false_easting=false_easting, false_northing=false_northing, globe=globe)
        self.threshold = 100000.0