from __future__ import division
from collections import OrderedDict
from contextlib import ExitStack
from distutils.version import LooseVersion
import math
import click
import snuggs
import rasterio
from rasterio.features import sieve
from rasterio.fill import fillnodata
from rasterio.windows import Window
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout
def _chunk_output(width, height, count, itemsize, mem_limit=1):
    """Divide the calculation output into chunks

    This function determines the chunk size such that an array of shape
    (chunk_size, chunk_size, count) with itemsize bytes per element
    requires no more than mem_limit megabytes of memory.

    Output chunks are described by rasterio Windows.

    Parameters
    ----------
    width : int
        Output width
    height : int
        Output height
    count : int
        Number of output bands
    itemsize : int
        Number of bytes per pixel
    mem_limit : int, default
        The maximum size in memory of a chunk array

    Returns
    -------
    sequence of Windows
    """
    max_pixels = mem_limit * 1000000.0 / itemsize * count
    chunk_size = int(math.floor(math.sqrt(max_pixels)))
    ncols = int(math.ceil(width / chunk_size))
    nrows = int(math.ceil(height / chunk_size))
    chunk_windows = []
    for col in range(ncols):
        col_offset = col * chunk_size
        w = min(chunk_size, width - col_offset)
        for row in range(nrows):
            row_offset = row * chunk_size
            h = min(chunk_size, height - row_offset)
            chunk_windows.append(((row, col), Window(col_offset, row_offset, w, h)))
    return chunk_windows