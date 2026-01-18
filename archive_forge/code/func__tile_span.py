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
def _tile_span(self, tile_matrix, meters_per_unit):
    pixel_span = tile_matrix.scaledenominator * (METERS_PER_PIXEL / meters_per_unit)
    tile_span_x = tile_matrix.tilewidth * pixel_span
    tile_span_y = tile_matrix.tileheight * pixel_span
    return (tile_span_x, tile_span_y)