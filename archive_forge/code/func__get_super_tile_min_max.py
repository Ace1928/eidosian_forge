from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def _get_super_tile_min_max(tile_info, load_data_func, rasterize_func):
    tile_size = tile_info['tile_size']
    df = load_data_func(tile_info['x_range'], tile_info['y_range'])
    agg = rasterize_func(df, x_range=tile_info['x_range'], y_range=tile_info['y_range'], height=tile_size, width=tile_size)
    return agg