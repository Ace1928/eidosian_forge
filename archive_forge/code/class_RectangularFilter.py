from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class RectangularFilter(Filter):
    """
    Rectangular filter class.

    Create a rectangular shaped filter with length `stop`, height 1 (unless
    normalized) with indices < `start` set to 0.

    Parameters
    ----------
    start : int
        Start bin of the filter.
    stop : int
        Stop bin of the filter.
    norm : bool, optional
        Normalize the area of the filter to 1.

    """

    def __init__(self, start, stop, norm=False):
        pass

    def __new__(cls, start, stop, norm=False):
        if start >= stop:
            raise ValueError('`start` must be smaller than `stop`')
        length = stop - start
        data = np.ones(length, dtype=np.float)
        return Filter.__new__(cls, data, start, norm)

    @classmethod
    def band_bins(cls, bins, overlap=False):
        """
        Yields start and stop bins and normalization info for creation of
        rectangular filters.

        Parameters
        ----------
        bins : list or numpy array
            Crossover bins of filters.
        overlap : bool, optional
            Filters should overlap.

        Yields
        ------
        start : int
            Start bin of the filter.
        stop : int
            Stop bin of the filter.

        """
        if len(bins) < 2:
            raise ValueError('not enough bins to create a RectangularFilter')
        if overlap:
            raise NotImplementedError('please implement if needed!')
        index = 0
        while index + 2 <= len(bins):
            start, stop = bins[index:index + 2]
            yield (start, stop)
            index += 1