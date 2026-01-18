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
def _interpolate_alpha(data, total, mask, how, alpha, span, min_alpha, rescale_discrete_levels):
    if cupy and isinstance(data, cupy.ndarray):
        from ._cuda_utils import interp, masked_clip_2d
        array_module = cupy
    else:
        from ._cpu_utils import masked_clip_2d
        interp = np.interp
        array_module = np
    if span is None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            offset = np.nanmin(total)
        if total.dtype.kind == 'u' and offset == 0:
            mask = mask | (total == 0)
            if not np.all(mask):
                offset = total[total > 0].min()
            total = np.where(~mask, total, np.nan)
        a_scaled = _normalize_interpolate_how(how)(total - offset, mask)
        discrete_levels = None
        if isinstance(a_scaled, (list, tuple)):
            a_scaled, discrete_levels = a_scaled
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
            norm_span = [np.nanmin(a_scaled).item(), np.nanmax(a_scaled).item()]
        if rescale_discrete_levels and discrete_levels is not None:
            norm_span = _rescale_discrete_levels(discrete_levels, norm_span)
    else:
        if how == 'eq_hist':
            raise ValueError('span is not (yet) valid to use with eq_hist')
        offset = np.array(span, dtype=data.dtype)[0]
        if total.dtype.kind == 'u' and np.nanmin(total) == 0:
            mask = mask | (total <= 0)
            total = np.where(~mask, total, np.nan)
        masked_clip_2d(total, mask, *span)
        a_scaled = _normalize_interpolate_how(how)(total - offset, mask)
        if isinstance(a_scaled, (list, tuple)):
            a_scaled = a_scaled[0]
        norm_span = _normalize_interpolate_how(how)([0, span[1] - span[0]], 0)
        if isinstance(norm_span, (list, tuple)):
            norm_span = norm_span[0]
    norm_span = array_module.hstack(norm_span)
    a_float = interp(a_scaled, norm_span, array_module.array([min_alpha, alpha]), left=0, right=255)
    a = np.nan_to_num(a_float, copy=False).astype(np.uint8)
    return a