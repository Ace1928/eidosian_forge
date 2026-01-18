import json
import logging
import os.path
import click
import cligj
import rasterio
from rasterio.rio import options
from rasterio.rio.helpers import write_features
from rasterio.warp import transform_bounds
def _normalize_bounds(self, bounds):
    if self._geographic:
        bounds = transform_bounds(self._src.crs, 'EPSG:4326', *bounds)
    if self._precision >= 0:
        bounds = (round(v, self._precision) for v in bounds)
    return bounds