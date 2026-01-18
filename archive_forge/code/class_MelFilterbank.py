from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class MelFilterbank(Filterbank):
    """
    Mel filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    num_bands : int, optional
        Number of filter bands.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.

    Notes
    -----
    Because of rounding and mapping of frequencies to bins and back to
    frequencies, the actual minimum, maximum and center frequencies do not
    necessarily match the parameters given.

    """
    NUM_BANDS = 40
    FMIN = 20.0
    FMAX = 17000.0
    NORM_FILTERS = True
    UNIQUE_FILTERS = True

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        pass

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        frequencies = mel_frequencies(num_bands + 2, fmin, fmax)
        bins = frequencies2bins(frequencies, bin_frequencies, unique_bins=unique_filters)
        filters = TriangularFilter.filters(bins, norm=norm_filters, overlap=True)
        return cls.from_filters(filters, bin_frequencies)