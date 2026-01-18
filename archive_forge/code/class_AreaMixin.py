import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class AreaMixin:

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        vdims = element.vdims[:2]
        vdim = vdims[0].name
        if len(vdims) > 1:
            new_range = {}
            for r in ranges[vdim]:
                if r != 'values':
                    new_range[r] = util.max_range([ranges[vd.name][r] for vd in vdims])
            ranges[vdim] = new_range
        else:
            s0, s1 = ranges[vdim]['soft']
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[vdim]['soft'] = (s0, s1)
        return super().get_extents(element, ranges, range_type)