import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def _fallback_proj_and_srs(self):
    """
        Return a :class:`cartopy.crs.Projection` and corresponding
        SRS string in which the WMS service can supply the requested
        layers.

        """
    contents = self.service.contents
    for proj, srs_list in _CRS_TO_OGC_SRS.items():
        for srs in srs_list:
            srs_OK = all((srs.lower() in map(str.lower, contents[layer].crsOptions) for layer in self.layers))
            if srs_OK:
                return (proj, srs)
    raise ValueError('The requested layers are not available in a known SRS.')