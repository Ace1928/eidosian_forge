from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class RectangularFilterbank(Filterbank):
    """
    Rectangular filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    crossover_frequencies : list or numpy array
        Crossover frequencies of the bands [Hz].
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.

    """

    def __init__(self, bin_frequencies, crossover_frequencies, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS):
        pass

    def __new__(cls, bin_frequencies, crossover_frequencies, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS):
        fb = np.zeros((len(bin_frequencies), len(crossover_frequencies) + 1), dtype=FILTER_DTYPE)
        corner_frequencies = np.r_[fmin, crossover_frequencies, fmax]
        corner_bins = frequencies2bins(corner_frequencies, bin_frequencies, unique_bins=unique_filters)
        for i in range(len(corner_bins) - 1):
            fb[corner_bins[i]:corner_bins[i + 1], i] = 1
        if norm_filters:
            band_sum = np.sum(fb, axis=0)
            band_sum[band_sum == 0] = 1
            fb /= band_sum
        obj = Filterbank.__new__(cls, fb, bin_frequencies)
        obj.crossover_frequencies = bins2frequencies(corner_bins[1:-1], bin_frequencies)
        return obj