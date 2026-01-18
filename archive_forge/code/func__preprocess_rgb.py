import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
def _preprocess_rgb(self, element):
    rgbarray = np.dstack([element.dimension_values(vd, flat=False) for vd in element.vdims])
    if rgbarray.dtype.kind == 'f':
        rgbarray = rgbarray * 255
    return tf.Image(self.uint8_to_uint32(rgbarray.astype('uint8')))