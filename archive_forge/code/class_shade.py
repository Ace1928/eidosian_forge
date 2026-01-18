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
class shade(LinkableOperation):
    """
    shade applies a normalization function followed by colormapping to
    an Image or NdOverlay of Images, returning an RGB Element.
    The data must be in the form of a 2D or 3D DataArray, but NdOverlays
    of 2D Images will be automatically converted to a 3D array.

    In the 2D case data is normalized and colormapped, while a 3D
    array representing categorical aggregates will be supplied a color
    key for each category. The colormap (cmap) for the 2D case may be
    supplied as an Iterable or a Callable.
    """
    alpha = param.Integer(default=255, bounds=(0, 255), doc='\n        Value between 0 - 255 representing the alpha value to use for\n        colormapped pixels that contain data (i.e. non-NaN values).\n        Regardless of this value, ``NaN`` values are set to be fully\n        transparent when doing colormapping.')
    cmap = param.ClassSelector(class_=(Iterable, Callable, dict), doc="\n        Iterable or callable which returns colors as hex colors\n        or web color names (as defined by datashader), to be used\n        for the colormap of single-layer datashader output.\n        Callable type must allow mapping colors between 0 and 1.\n        The default value of None reverts to Datashader's default\n        colormap.")
    color_key = param.ClassSelector(class_=(Iterable, Callable, dict), doc='\n        Iterable or callable that returns colors as hex colors, to\n        be used for the color key of categorical datashader output.\n        Callable type must allow mapping colors for supplied values\n        between 0 and 1.')
    cnorm = param.ClassSelector(default='eq_hist', class_=(str, Callable), doc="\n        The normalization operation applied before colormapping.\n        Valid options include 'linear', 'log', 'eq_hist', 'cbrt',\n        and any valid transfer function that accepts data, mask, nbins\n        arguments.")
    clims = param.NumericTuple(default=None, length=2, doc='\n        Min and max data values to use for colormap interpolation, when\n        wishing to override autoranging.\n        ')
    min_alpha = param.Number(default=40, bounds=(0, 255), doc='\n        The minimum alpha value to use for non-empty pixels when doing\n        colormapping, in [0, 255].  Use a higher value to avoid\n        undersaturation, i.e. poorly visible low-value datapoints, at\n        the expense of the overall dynamic range..')
    rescale_discrete_levels = param.Boolean(default=True, doc="\n        If ``cnorm='eq_hist`` and there are only a few discrete values,\n        then ``rescale_discrete_levels=True`` (the default) decreases\n        the lower limit of the autoranged span so that the values are\n        rendering towards the (more visible) top of the ``cmap`` range,\n        thus avoiding washout of the lower values.  Has no effect if\n        ``cnorm!=`eq_hist``. Set this value to False if you need to\n        match historical unscaled behavior, prior to HoloViews 1.14.4.")

    @classmethod
    def concatenate(cls, overlay):
        """
        Concatenates an NdOverlay of Image types into a single 3D
        xarray Dataset.
        """
        if not isinstance(overlay, NdOverlay):
            raise ValueError('Only NdOverlays can be concatenated')
        xarr = xr.concat([v.data.transpose() for v in overlay.values()], pd.Index(overlay.keys(), name=overlay.kdims[0].name))
        params = dict(get_param_values(overlay.last), vdims=overlay.last.vdims, kdims=overlay.kdims + overlay.last.kdims)
        return Dataset(xarr.transpose(), datatype=['xarray'], **params)

    @classmethod
    def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

    @classmethod
    def uint32_to_uint8_xr(cls, img):
        """
        Cast uint32 xarray DataArray to 4 uint8 channels.
        """
        new_array = img.values.view(dtype=np.uint8).reshape(img.shape + (4,))
        coords = dict(list(img.coords.items()) + [('band', [0, 1, 2, 3])])
        return xr.DataArray(new_array, coords=coords, dims=img.dims + ('band',))

    @classmethod
    def rgb2hex(cls, rgb):
        """
        Convert RGB(A) tuple to hex.
        """
        if len(rgb) > 3:
            rgb = rgb[:-1]
        return '#{:02x}{:02x}{:02x}'.format(*(int(v * 255) for v in rgb))

    @classmethod
    def to_xarray(cls, element):
        if issubclass(element.interface, XArrayInterface):
            return element
        data = tuple((element.dimension_values(kd, expanded=False) for kd in element.kdims))
        vdims = list(element.vdims)
        element.vdims[:] = [vd.clone(nodata=None) for vd in element.vdims]
        try:
            data += tuple((element.dimension_values(vd, flat=False) for vd in element.vdims))
        finally:
            element.vdims[:] = vdims
        dtypes = [dt for dt in element.datatype if dt != 'xarray']
        return element.clone(data, datatype=['xarray'] + dtypes, bounds=element.bounds, xdensity=element.xdensity, ydensity=element.ydensity)

    def _process(self, element, key=None):
        element = element.map(self.to_xarray, Image)
        if isinstance(element, NdOverlay):
            bounds = element.last.bounds
            xdensity = element.last.xdensity
            ydensity = element.last.ydensity
            element = self.concatenate(element)
        elif isinstance(element, Overlay):
            return element.map(partial(shade._process, self), [Element])
        else:
            xdensity = element.xdensity
            ydensity = element.ydensity
            bounds = element.bounds
        kdims = element.kdims
        if isinstance(element, ImageStack):
            vdim = element.vdims
            array = element.data
            if hasattr(array, 'to_array'):
                array = array.to_array('z')
            array = array.transpose(*[kdim.name for kdim in kdims], ...)
        else:
            vdim = element.vdims[0].name
            array = element.data[vdim]
        shade_opts = dict(how=self.p.cnorm, min_alpha=self.p.min_alpha, alpha=self.p.alpha)
        if ds_version >= Version('0.14.0'):
            shade_opts['rescale_discrete_levels'] = self.p.rescale_discrete_levels
        if element.ndims > 2 or isinstance(element, ImageStack):
            kdims = element.kdims if isinstance(element, ImageStack) else element.kdims[1:]
            categories = array.shape[-1]
            if not self.p.color_key:
                pass
            elif isinstance(self.p.color_key, dict):
                shade_opts['color_key'] = self.p.color_key
            elif isinstance(self.p.color_key, Iterable):
                shade_opts['color_key'] = [c for _, c in zip(range(categories), self.p.color_key)]
            else:
                colors = [self.p.color_key(s) for s in np.linspace(0, 1, categories)]
                shade_opts['color_key'] = map(self.rgb2hex, colors)
        elif not self.p.cmap:
            pass
        elif isinstance(self.p.cmap, Callable):
            colors = [self.p.cmap(s) for s in np.linspace(0, 1, 256)]
            shade_opts['cmap'] = map(self.rgb2hex, colors)
        elif isinstance(self.p.cmap, str):
            if self.p.cmap.startswith('#') or self.p.cmap in color_lookup:
                shade_opts['cmap'] = self.p.cmap
            else:
                from ..plotting.util import process_cmap
                shade_opts['cmap'] = process_cmap(self.p.cmap)
        else:
            shade_opts['cmap'] = self.p.cmap
        if self.p.clims:
            shade_opts['span'] = self.p.clims
        elif ds_version > Version('0.5.0') and self.p.cnorm != 'eq_hist':
            shade_opts['span'] = element.range(vdim)
        params = dict(get_param_values(element), kdims=kdims, bounds=bounds, vdims=RGB.vdims[:], xdensity=xdensity, ydensity=ydensity)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')
            if np.isnan(array.data).all():
                xd, yd = kdims[:2]
                arr = np.zeros(array.data.shape[:2] + (4,), dtype=np.uint8)
                coords = {xd.name: element.data.coords[xd.name], yd.name: element.data.coords[yd.name], 'band': [0, 1, 2, 3]}
                img = xr.DataArray(arr, coords=coords, dims=(yd.name, xd.name, 'band'))
                return RGB(img, **params)
            else:
                img = tf.shade(array, **shade_opts)
        return RGB(self.uint32_to_uint8_xr(img), **params)