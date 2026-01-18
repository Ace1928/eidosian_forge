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
def _select_tiles(self, tile_matrix, tile_matrix_limits, tile_span_x, tile_span_y, extent):
    min_x, max_x, min_y, max_y = extent
    matrix_min_x, matrix_max_y = tile_matrix.topleftcorner
    epsilon = 1e-06
    min_col = int((min_x - matrix_min_x) / tile_span_x + epsilon)
    max_col = int((max_x - matrix_min_x) / tile_span_x - epsilon)
    min_row = int((matrix_max_y - max_y) / tile_span_y + epsilon)
    max_row = int((matrix_max_y - min_y) / tile_span_y - epsilon)
    min_col = max(min_col, 0)
    max_col = min(max_col, tile_matrix.matrixwidth - 1)
    min_row = max(min_row, 0)
    max_row = min(max_row, tile_matrix.matrixheight - 1)
    if tile_matrix_limits:
        min_col = max(min_col, tile_matrix_limits.mintilecol)
        max_col = min(max_col, tile_matrix_limits.maxtilecol)
        min_row = max(min_row, tile_matrix_limits.mintilerow)
        max_row = min(max_row, tile_matrix_limits.maxtilerow)
    return (min_col, max_col, min_row, max_row)