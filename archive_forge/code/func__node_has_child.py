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
@staticmethod
def _node_has_child(node, find_str):
    """
        Return whether `node` contains (at any sub-level), a node with name
        equal to `find_str`.

        """
    element = node.find(find_str)
    return element is not None