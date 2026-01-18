import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class GeomMixin:

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        """
        Use first two key dimensions to set names, and all four
        to set the data range.
        """
        kdims = element.kdims
        for kdim0, kdim1 in zip([kdims[i].name for i in range(2)], [kdims[i].name for i in range(2, 4)]):
            new_range = {}
            for kdim in [kdim0, kdim1]:
                for r in ranges[kdim]:
                    if r == 'factors':
                        new_range[r] = list(util.unique_iterator(list(ranges[kdim0][r]) + list(ranges[kdim1][r])))
                    else:
                        new_range[r] = util.max_range([ranges[kd][r] for kd in [kdim0, kdim1]])
            ranges[kdim0] = new_range
            ranges[kdim1] = new_range
        return super().get_extents(element, ranges, range_type)