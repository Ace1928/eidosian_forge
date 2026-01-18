from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
def _merge_tiles(tiles):
    """Return a single image, merging the given images."""
    if not tiles:
        raise ValueError('A non-empty list of tiles should be provided to merge.')
    xset = [set(x) for i, x, y, _ in tiles]
    yset = [set(y) for i, x, y, _ in tiles]
    xs = xset[0]
    xs.update(*xset[1:])
    ys = yset[0]
    ys.update(*yset[1:])
    xs = sorted(xs)
    ys = sorted(ys)
    other_len = tiles[0][0].shape[2:]
    img = np.zeros((len(ys), len(xs)) + other_len, dtype=np.uint8) - 1
    for tile_img, x, y, origin in tiles:
        y_first, y_last = (y[0], y[-1])
        yi0, yi1 = np.where((y_first == ys) | (y_last == ys))[0]
        if origin == 'upper':
            yi0 = tile_img.shape[0] - yi0 - 1
            yi1 = tile_img.shape[0] - yi1 - 1
        start, stop, step = (yi0, yi1, 1 if yi0 < yi1 else -1)
        if step == 1 and stop == img.shape[0] - 1:
            stop = None
        elif step == -1 and stop == 0:
            stop = None
        else:
            stop += step
        y_slice = slice(start, stop, step)
        xi0, xi1 = np.where((x[0] == xs) | (x[-1] == xs))[0]
        start, stop, step = (xi0, xi1, 1 if xi0 < xi1 else -1)
        if step == 1 and stop == img.shape[1] - 1:
            stop = None
        elif step == -1 and stop == 0:
            stop = None
        else:
            stop += step
        x_slice = slice(start, stop, step)
        img_slice = (y_slice, x_slice, Ellipsis)
        if origin == 'lower':
            tile_img = tile_img[::-1, :]
        img[img_slice] = tile_img
    return (img, [min(xs), max(xs), min(ys), max(ys)], 'lower')