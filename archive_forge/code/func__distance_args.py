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
@classmethod
def _distance_args(cls, element, x_range, y_range, pixels):
    ycount, xcount = element.interface.shape(element, gridded=True)
    x_delta = abs(x_range[1] - x_range[0]) / xcount
    y_delta = abs(y_range[1] - y_range[0]) / ycount
    return (x_delta * pixels, y_delta * pixels)