from __future__ import annotations
from collections.abc import Iterator
from io import BytesIO
import warnings
import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
import dask.array as da
from PIL.Image import fromarray
from datashader.colors import rgb, Sets1to3
from datashader.utils import nansum_missing, ngjit
def _apply_discrete_colorkey(agg, color_key, alpha, name, color_baseline):
    if cupy and isinstance(agg.data, cupy.ndarray):
        module = cupy
        array = cupy.array
    else:
        module = np
        array = np.array
    if not agg.ndim == 2:
        raise ValueError('agg must be 2D')
    if color_key is None or not isinstance(color_key, dict):
        raise ValueError('Color key must be provided as a dictionary')
    agg_data = agg.data
    if isinstance(agg_data, da.Array):
        agg_data = agg_data.compute()
    cats = color_key.keys()
    colors = [rgb(color_key[c]) for c in cats]
    rs, gs, bs = map(array, zip(*colors))
    data = module.empty_like(agg_data) * module.nan
    r = module.zeros_like(data, dtype=module.uint8)
    g = module.zeros_like(data, dtype=module.uint8)
    b = module.zeros_like(data, dtype=module.uint8)
    r2 = module.zeros_like(data, dtype=module.uint8)
    g2 = module.zeros_like(data, dtype=module.uint8)
    b2 = module.zeros_like(data, dtype=module.uint8)
    for i, c in enumerate(cats):
        value_mask = agg_data == c
        data[value_mask] = 1
        r2[value_mask] = rs[i]
        g2[value_mask] = gs[i]
        b2[value_mask] = bs[i]
    color_data = data.copy()
    baseline = module.nanmin(color_data) if color_baseline is None else color_baseline
    with np.errstate(invalid='ignore'):
        if baseline > 0:
            color_data -= baseline
        elif baseline < 0:
            color_data += -baseline
        if color_data.dtype.kind != 'u' and color_baseline is not None:
            color_data[color_data < 0] = 0
    color_data[module.isnan(data)] = 0
    if not color_data.any():
        r[:] = r2
        g[:] = g2
        b[:] = b2
    missing_colors = color_data == 0
    r = module.where(missing_colors, r2, r)
    g = module.where(missing_colors, g2, g)
    b = module.where(missing_colors, b2, b)
    a = np.where(np.isnan(data), 0, alpha).astype(np.uint8)
    values = module.dstack([r, g, b, a]).view(module.uint32).reshape(a.shape)
    if cupy and isinstance(agg.data, cupy.ndarray):
        values = cupy.asnumpy(values)
    return Image(values, dims=agg.dims, coords=agg.coords, name=name)