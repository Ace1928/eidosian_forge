from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class LogarithmicFilterbank(Filterbank):
    """
    Logarithmic filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    num_bands : int, optional
        Number of filter bands (per octave).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.
    bands_per_octave : bool, optional
        Indicates whether `num_bands` is given as number of bands per octave
        ('True', default) or as an absolute number of bands ('False').

    Notes
    -----
    `num_bands` sets either the number of bands per octave or the total number
    of bands, depending on the setting of `bands_per_octave`. `num_bands` is
    used to set also the number of bands per octave to keep the argument for
    all classes the same. If 12 bands per octave are used, a filterbank with
    semitone spacing is created.

    """
    NUM_BANDS_PER_OCTAVE = 12

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE, fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        pass

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE, fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        if bands_per_octave:
            num_bands_per_octave = num_bands
            frequencies = log_frequencies(num_bands, fmin, fmax, fref)
            bins = frequencies2bins(frequencies, bin_frequencies, unique_bins=unique_filters)
        else:
            raise NotImplementedError("please implement `num_bands` with `bands_per_octave` set to 'False' for LogarithmicFilterbank")
        filters = TriangularFilter.filters(bins, norm=norm_filters, overlap=True)
        obj = cls.from_filters(filters, bin_frequencies)
        obj.fref = fref
        obj.num_bands_per_octave = num_bands_per_octave
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.num_bands_per_octave = getattr(obj, 'num_bands_per_octave', self.NUM_BANDS_PER_OCTAVE)
        self.fref = getattr(obj, 'fref', A4)