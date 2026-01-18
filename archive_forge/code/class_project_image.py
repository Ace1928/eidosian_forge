import logging
import sys
import param
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from holoviews.core.data import MultiInterface
from holoviews.core.util import cartesian_product, get_param_values
from holoviews.operation import Operation
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from ..data import GeoPandasInterface
from ..element import (Image, Shape, Polygons, Path, Points, Contours,
from ..util import (
class project_image(_project_operation):
    """
    Projects an geoviews Image to the specified projection,
    returning a regular HoloViews Image type. Works by
    regridding the data along projected bounds. Only supports
    rectangular projections.
    """
    fast = param.Boolean(default=False, doc='\n        Whether to enable fast reprojection with (much) better\n        performance but poorer handling in polar regions.')
    width = param.Integer(default=None, doc='\n        Width of the reprojectd Image')
    height = param.Integer(default=None, doc='\n        Height of the reprojected Image')
    link_inputs = param.Boolean(default=True, doc='\n        By default, the link_inputs parameter is set to True so that\n        when applying project_image, backends that support linked streams\n        update RangeXY streams on the inputs of the operation.')
    supported_types = [Image, RGB]

    def _process(self, img, key=None):
        from cartopy.img_transform import warp_array
        if self.p.fast:
            return self._fast_process(img, key)
        proj = self.p.projection
        x0, x1 = img.range(0, dimension_range=False)
        y0, y1 = img.range(1, dimension_range=False)
        yn, xn = img.interface.shape(img, gridded=True)[:2]
        px0, py0, px1, py1 = project_extents((x0, y0, x1, y1), img.crs, proj)
        eps = sys.float_info.epsilon
        src_extent = tuple((e + v if e == 0 else e for e, v in zip((x0, x1, y0, y1), (eps, -eps, eps, -eps))))
        tgt_extent = (px0, px1, py0, py1)
        if img.crs == proj and np.isclose(src_extent, tgt_extent).all():
            return img
        arrays = []
        for vd in img.vdims:
            arr = img.dimension_values(vd, flat=False)
            if arr.size:
                projected, _ = warp_array(arr, proj, img.crs, (xn, yn), src_extent, tgt_extent)
            else:
                projected = arr
            arrays.append(projected)
        if xn == 0 or yn == 0:
            return img.clone([], bounds=tgt_extent, crs=proj)
        xunit = (tgt_extent[1] - tgt_extent[0]) / float(xn) / 2.0
        yunit = (tgt_extent[3] - tgt_extent[2]) / float(yn) / 2.0
        xs = np.linspace(tgt_extent[0] + xunit, tgt_extent[1] - xunit, xn)
        ys = np.linspace(tgt_extent[2] + yunit, tgt_extent[3] - yunit, yn)
        return img.clone((xs, ys) + tuple(arrays), bounds=None, kdims=img.kdims, vdims=img.vdims, crs=proj, xdensity=None, ydensity=None)

    def _fast_process(self, element, key=None):
        from cartopy.img_transform import _determine_bounds
        proj = self.p.projection
        if proj == element.crs:
            return element
        h, w = element.interface.shape(element, gridded=True)[:2]
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)
        if isinstance(element, RGB):
            rgb = element.rgb
            array = np.dstack([np.flipud(rgb.dimension_values(d, flat=False)) for d in rgb.vdims])
        else:
            array = element.dimension_values(2, flat=False)
        x0, y0, x1, y1 = element.bounds.lbrt()
        width = int(w) if self.p.width is None else self.p.width
        height = int(h) if self.p.height is None else self.p.height
        bounds = _determine_bounds(xs, ys, element.crs)
        yb = bounds['y']
        resampled = []
        xvalues = []
        for xb in bounds['x']:
            px0, py0, px1, py1 = project_extents((xb[0], yb[0], xb[1], yb[1]), element.crs, proj)
            if len(bounds['x']) > 1:
                xfraction = (xb[1] - xb[0]) / (x1 - x0)
                fraction_width = int(width * xfraction)
            else:
                fraction_width = width
            xs = np.linspace(px0, px1, fraction_width)
            ys = np.linspace(py0, py1, height)
            cxs, cys = cartesian_product([xs, ys])
            pxs, pys, _ = element.crs.transform_points(proj, np.asarray(cxs), np.asarray(cys)).T
            icxs = ((pxs - x0) / (x1 - x0) * w).astype(int)
            icys = ((pys - y0) / (y1 - y0) * h).astype(int)
            xvalues.append(xs)
            icxs[icxs < 0] = 0
            icys[icys < 0] = 0
            icxs[icxs >= w] = w - 1
            icys[icys >= h] = h - 1
            resampled_arr = array[icys, icxs]
            if isinstance(element, RGB):
                nvdims = len(element.vdims)
                resampled_arr = resampled_arr.reshape((fraction_width, height, nvdims)).transpose([1, 0, 2])
            else:
                resampled_arr = resampled_arr.reshape((fraction_width, height)).T
            resampled.append(resampled_arr)
        xs = np.concatenate(xvalues[::-1])
        resampled = np.hstack(resampled[::-1])
        datatypes = [element.interface.datatype, 'xarray', 'grid']
        data = (xs, ys)
        for i in range(len(element.vdims)):
            if resampled.ndim > 2:
                data = data + (resampled[::-1, :, i],)
            else:
                data = data + (resampled,)
        return element.clone(data, crs=proj, bounds=None, datatype=datatypes)