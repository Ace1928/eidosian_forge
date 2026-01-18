import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class BarsMixin:

    def _get_axis_dims(self, element):
        if element.ndims > 1 and (not (self.stacked or not self.multi_level)):
            xdims = element.kdims
        else:
            xdims = element.kdims[0]
        return (xdims, element.vdims[0])

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        """
        Make adjustments to plot extents by computing
        stacked bar heights, adjusting the bar baseline
        and forcing the x-axis to be categorical.
        """
        if self.batched:
            overlay = self.current_frame
            element = Bars(overlay.table(), kdims=element.kdims + overlay.kdims, vdims=element.vdims)
            for kd in overlay.kdims:
                ranges[kd.name]['combined'] = overlay.range(kd)
        vdim = element.vdims[0].name
        s0, s1 = ranges[vdim]['soft']
        s0 = min(s0, 0) if util.isfinite(s0) else 0
        s1 = max(s1, 0) if util.isfinite(s1) else 0
        ranges[vdim]['soft'] = (s0, s1)
        if range_type not in ('combined', 'data'):
            return super().get_extents(element, ranges, range_type, ydim=element.vdims[0])
        xdim = element.kdims[0]
        if self.stacked:
            ds = Dataset(element)
            pos_range = ds.select(**{vdim: (0, None)}).aggregate(xdim, function=np.sum).range(vdim)
            neg_range = ds.select(**{vdim: (None, 0)}).aggregate(xdim, function=np.sum).range(vdim)
            y0, y1 = util.max_range([pos_range, neg_range])
        else:
            y0, y1 = ranges[vdim]['combined']
        if range_type == 'data':
            return ('', y0, '', y1)
        padding = 0 if self.overlaid else self.padding
        _, ypad, _ = get_axis_padding(padding)
        y0, y1 = util.dimension_range(y0, y1, ranges[vdim]['hard'], ranges[vdim]['soft'], ypad, self.logy)
        y0, y1 = util.dimension_range(y0, y1, self.ylim, (None, None))
        return ('', y0, '', y1)

    def _get_coords(self, element, ranges, as_string=True):
        """
        Get factors for categorical axes.
        """
        gdim = None
        sdim = None
        if element.ndims == 1:
            pass
        elif not self.stacked:
            gdim = element.get_dimension(1)
        else:
            sdim = element.get_dimension(1)
        xdim, ydim = element.dimensions()[:2]
        xvals = None
        if xdim.values:
            xvals = xdim.values
        if gdim and (not sdim):
            if not xvals and (not gdim.values):
                xvals, gvals = categorical_aggregate2d._get_coords(element)
            else:
                if gdim.values:
                    gvals = gdim.values
                elif ranges.get(gdim.name, {}).get('factors') is not None:
                    gvals = ranges[gdim.name]['factors']
                else:
                    gvals = element.dimension_values(gdim, False)
                gvals = np.asarray(gvals)
                if xvals:
                    pass
                elif ranges.get(xdim.name, {}).get('factors') is not None:
                    xvals = ranges[xdim.name]['factors']
                else:
                    xvals = element.dimension_values(0, False)
                xvals = np.asarray(xvals)
            c_is_str = xvals.dtype.kind in 'SU' or not as_string
            g_is_str = gvals.dtype.kind in 'SU' or not as_string
            xvals = [x if c_is_str else xdim.pprint_value(x) for x in xvals]
            gvals = [g if g_is_str else gdim.pprint_value(g) for g in gvals]
            return (xvals, gvals)
        else:
            if xvals:
                pass
            elif ranges.get(xdim.name, {}).get('factors') is not None:
                xvals = ranges[xdim.name]['factors']
            else:
                xvals = element.dimension_values(0, False)
            xvals = np.asarray(xvals)
            c_is_str = xvals.dtype.kind in 'SU' or not as_string
            xvals = [x if c_is_str else xdim.pprint_value(x) for x in xvals]
            return (xvals, None)