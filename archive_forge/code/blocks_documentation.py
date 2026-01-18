import json
import logging
import os.path
import click
import cligj
import rasterio
from rasterio.rio import options
from rasterio.rio.helpers import write_features
from rasterio.warp import transform_bounds
Export raster dataset windows to GeoJSON polygon features.

        Parameters
        ----------
        dataset : a dataset object opened in 'r' mode
            Source dataset
        bidx : int
            Extract windows from this band
        precision : int, optional
            Coordinate precision
        geographic : bool, optional
            Reproject geometries to ``EPSG:4326`` if ``True``

        Yields
        ------
        dict
            GeoJSON polygon feature
        