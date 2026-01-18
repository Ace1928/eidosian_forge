from collections import UserDict
import warnings
from rasterio.dtypes import uint8
class DefaultGTiffProfile(Profile):
    """Tiled, band-interleaved, LZW-compressed, 8-bit GTiff."""
    defaults = {'driver': 'GTiff', 'interleave': 'band', 'tiled': True, 'blockxsize': 256, 'blockysize': 256, 'compress': 'lzw', 'nodata': 0, 'dtype': uint8}