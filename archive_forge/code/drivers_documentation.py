import os
from rasterio._base import _raster_driver_extensions
from rasterio.env import GDALVersion, ensure_env
Returns True if driver `name` and `mode` are blacklisted.