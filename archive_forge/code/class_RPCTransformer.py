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
class RPCTransformer(RPCTransformerBase, GDALTransformerBase):
    """
    Class related to Rational Polynomial Coeffecients (RPCs) based
    coordinate transformations.

    Uses GDALCreateRPCTransformer and GDALRPCTransform for computations. Options
    for GDALCreateRPCTransformer may be passed using `rpc_options`.
    Ensure that GDAL transformer objects are destroyed by calling `close()`
    method or using context manager interface.

    """

    def __init__(self, rpcs, **rpc_options):
        if not isinstance(rpcs, (RPC, dict)):
            raise ValueError('RPCTransformer requires RPC')
        super().__init__(rpcs, **rpc_options)

    def __repr__(self):
        return '<{} RPCTransformer>'.format(self.closed and 'closed' or 'open')