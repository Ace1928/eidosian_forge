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
def _get_bands(inputs, sources, d, i=None):
    """Get a rasterio.Band object from calc's inputs"""
    idx = d if d in dict(inputs) else int(d) - 1
    src = sources[idx]
    return rasterio.band(src, i) if i else [rasterio.band(src, j) for j in src.indexes]