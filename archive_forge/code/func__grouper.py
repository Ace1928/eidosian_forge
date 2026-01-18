from itertools import zip_longest
import numpy as np
from rasterio.enums import MaskFlags
from rasterio.windows import Window
from rasterio.transform import rowcol
def _grouper(iterable, n, fillvalue=None):
    """Collect data into non-overlapping fixed-length chunks or blocks"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)