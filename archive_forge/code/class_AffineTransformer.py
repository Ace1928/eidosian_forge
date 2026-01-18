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
class AffineTransformer(TransformerBase):
    """A pure Python class related to affine based coordinate transformations."""

    def __init__(self, affine_transform):
        super().__init__()
        if not isinstance(affine_transform, Affine):
            raise ValueError('Not an affine transform')
        self._transformer = affine_transform

    def _transform(self, xs, ys, zs, transform_direction):
        resxs = []
        resys = []
        if transform_direction is TransformDirection.forward:
            transform = self._transformer
        elif transform_direction is TransformDirection.reverse:
            transform = ~self._transformer
        for x, y in zip(xs, ys):
            resx, resy = transform * (x, y)
            resxs.append(resx)
            resys.append(resy)
        return (resxs, resys)

    def __repr__(self):
        return '<AffineTransformer>'