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
def _wmts_images(self, wmts, layer, matrix_set_name, extent, max_pixel_span):
    """
        Add images from the specified WMTS layer and matrix set to cover
        the specified extent at an appropriate resolution.

        The zoom level (aka. tile matrix) is chosen to give the lowest
        possible resolution which still provides the requested quality.
        If insufficient resolution is available, the highest available
        resolution is used.

        Parameters
        ----------
        wmts
            The owslib.wmts.WebMapTileService providing the tiles.
        layer
            The owslib.wmts.ContentMetadata (aka. layer) to draw.
        matrix_set_name
            The name of the matrix set to use.
        extent
            Tuple of (left, right, bottom, top) in Axes coordinates.
        max_pixel_span
            Preferred maximum pixel width or height in Axes coordinates.

        """
    tile_matrix_set = wmts.tilematrixsets[matrix_set_name]
    tile_matrices = tile_matrix_set.tilematrix.values()
    if tile_matrix_set.crs in METERS_PER_UNIT:
        meters_per_unit = METERS_PER_UNIT[tile_matrix_set.crs]
    elif ':EPSG:' in tile_matrix_set.crs:
        epsg_num = tile_matrix_set.crs.split(':')[-1]
        tms_crs = ccrs.epsg(int(epsg_num))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            crs_dict = tms_crs.to_dict()
            crs_units = crs_dict.get('units', '')
            if crs_units != 'm':
                raise ValueError('Only unit "m" implemented for EPSG projections.')
            meters_per_unit = 1
    else:
        raise ValueError(f'Unknown coordinate reference system string: {tile_matrix_set.crs}')
    tile_matrix = self._choose_matrix(tile_matrices, meters_per_unit, max_pixel_span)
    tile_span_x, tile_span_y = self._tile_span(tile_matrix, meters_per_unit)
    tile_matrix_set_links = getattr(layer, 'tilematrixsetlinks', None)
    if tile_matrix_set_links is None:
        tile_matrix_limits = None
    else:
        tile_matrix_set_link = tile_matrix_set_links[matrix_set_name]
        tile_matrix_limits = tile_matrix_set_link.tilematrixlimits.get(tile_matrix.identifier)
    min_col, max_col, min_row, max_row = self._select_tiles(tile_matrix, tile_matrix_limits, tile_span_x, tile_span_y, extent)
    tile_matrix_id = tile_matrix.identifier
    cache_by_wmts = WMTSRasterSource._shared_image_cache
    cache_by_layer_matrix = cache_by_wmts.setdefault(wmts, {})
    image_cache = cache_by_layer_matrix.setdefault((layer.id, tile_matrix_id), {})
    big_img = None
    n_rows = 1 + max_row - min_row
    n_cols = 1 + max_col - min_col
    ignore_out_of_range = tile_matrix_set_links is None
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            img_key = (row, col)
            img = image_cache.get(img_key)
            if img is None:
                try:
                    tile = wmts.gettile(layer=layer.id, tilematrixset=matrix_set_name, tilematrix=str(tile_matrix_id), row=str(row), column=str(col), **self.gettile_extra_kwargs)
                except owslib.util.ServiceException as exception:
                    if 'TileOutOfRange' in exception.message and ignore_out_of_range:
                        continue
                    raise exception
                img = Image.open(io.BytesIO(tile.read()))
                image_cache[img_key] = img
            if big_img is None:
                size = (img.size[0] * n_cols, img.size[1] * n_rows)
                big_img = Image.new('RGBA', size, (255, 255, 255, 255))
            top = (row - min_row) * tile_matrix.tileheight
            left = (col - min_col) * tile_matrix.tilewidth
            big_img.paste(img, (left, top))
    if big_img is None:
        img_extent = None
    else:
        matrix_min_x, matrix_max_y = tile_matrix.topleftcorner
        min_img_x = matrix_min_x + tile_span_x * min_col
        max_img_y = matrix_max_y - tile_span_y * min_row
        img_extent = (min_img_x, min_img_x + n_cols * tile_span_x, max_img_y - n_rows * tile_span_y, max_img_y)
    return (big_img, img_extent)