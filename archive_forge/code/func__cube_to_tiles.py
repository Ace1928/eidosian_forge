from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def _cube_to_tiles(grid, vmin, vmax):
    slices = grid.shape[0]
    rows, columns, image_width, image_height = _compute_tile_size(grid.shape)
    image_height = rows * grid.shape[1]
    data = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    grid_normalized = (grid * 1.0 - vmin) / (vmax - vmin)
    grid_normalized[~np.isfinite(grid_normalized)] = 0
    gradient = np.gradient(grid_normalized)
    with np.errstate(invalid='ignore'):
        gradient = gradient / np.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2)
    for y2d in range(rows):
        for x2d in range(columns):
            zindex = x2d + y2d * columns
            if zindex < slices:
                Im = grid_normalized[zindex]
                subdata = data[y2d * Im.shape[0]:(y2d + 1) * Im.shape[0], x2d * Im.shape[1]:(x2d + 1) * Im.shape[1]]
                subdata[..., 3] = (Im * 255).astype(np.uint8)
                for i in range(3):
                    subdata[..., i] = ((gradient[i][zindex] / 2.0 + 0.5) * 255).astype(np.uint8)
    tile_shape = (grid.shape[2], grid.shape[1])
    return (data, tile_shape, rows, columns, grid.shape[0])