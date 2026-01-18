from contextlib import ExitStack
from functools import partial
import math
import numpy as np
import warnings
from affine import Affine
from rasterio.env import env_ctx_if_needed
from rasterio._transform import (
from rasterio.enums import TransformDirection, TransformMethod
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC
from rasterio.errors import TransformError, RasterioDeprecationWarning
@staticmethod
def _ensure_arr_input(xs, ys, zs=None):
    """Ensure all input coordinates are mapped to array-like objects

        Raises
        ------
        TransformError
            If input coordinates are not all of the same length
        """
    try:
        xs, ys, zs = np.broadcast_arrays(xs, ys, 0 if zs is None else zs)
    except ValueError as error:
        raise TransformError('Input coordinates must be broadcastable to a 1d array') from error
    return (np.atleast_1d(xs), np.atleast_1d(ys), np.atleast_1d(zs))