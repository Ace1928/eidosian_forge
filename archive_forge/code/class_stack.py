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
class stack(Operation):
    """
    The stack operation allows compositing multiple RGB Elements using
    the defined compositing operator.
    """
    compositor = param.ObjectSelector(objects=['add', 'over', 'saturate', 'source'], default='over', doc='\n        Defines how the compositing operation combines the images')

    def uint8_to_uint32(self, element):
        img = np.dstack([element.dimension_values(d, flat=False) for d in element.vdims])
        if img.shape[2] == 3:
            alpha = np.ones(img.shape[:2])
            if img.dtype.name == 'uint8':
                alpha = (alpha * 255).astype('uint8')
            img = np.dstack([img, alpha])
        if img.dtype.name != 'uint8':
            img = (img * 255).astype(np.uint8)
        N, M, _ = img.shape
        return img.view(dtype=np.uint32).reshape((N, M))

    def _process(self, overlay, key=None):
        if not isinstance(overlay, CompositeOverlay):
            return overlay
        elif len(overlay) == 1:
            return overlay.last if isinstance(overlay, NdOverlay) else overlay.get(0)
        imgs = []
        for rgb in overlay:
            if not isinstance(rgb, RGB):
                raise TypeError("The stack operation expects elements of type RGB, not '%s'." % type(rgb).__name__)
            rgb = rgb.rgb
            dims = [kd.name for kd in rgb.kdims][::-1]
            coords = {kd.name: rgb.dimension_values(kd, False) for kd in rgb.kdims}
            imgs.append(tf.Image(self.uint8_to_uint32(rgb), coords=coords, dims=dims))
        try:
            imgs = xr.align(*imgs, join='exact')
        except ValueError as e:
            raise ValueError('RGB inputs to the stack operation could not be aligned; ensure they share the same grid sampling.') from e
        stacked = tf.stack(*imgs, how=self.p.compositor)
        arr = shade.uint32_to_uint8(stacked.data)[::-1]
        data = (coords[dims[1]], coords[dims[0]], arr[:, :, 0], arr[:, :, 1], arr[:, :, 2])
        if arr.shape[-1] == 4:
            data = data + (arr[:, :, 3],)
        return rgb.clone(data, datatype=[rgb.interface.datatype] + rgb.datatype)