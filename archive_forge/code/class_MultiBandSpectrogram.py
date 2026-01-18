from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class MultiBandSpectrogram(FilteredSpectrogram):
    """
    MultiBandSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    crossover_frequencies : list or numpy array
        List of crossover frequencies at which the `spectrogram` is split
        into multiple bands.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter bands of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated
        with these additional keyword arguments.

    Notes
    -----
    The MultiBandSpectrogram is implemented as a :class:`Spectrogram` which
    uses a :class:`.audio.filters.RectangularFilterbank` to combine multiple
    frequency bins.

    """

    def __init__(self, spectrogram, crossover_frequencies, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        pass

    def __new__(cls, spectrogram, crossover_frequencies, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        from .filters import RectangularFilterbank
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        filterbank = RectangularFilterbank(spectrogram.bin_frequencies, crossover_frequencies, fmin=fmin, fmax=fmax, norm_filters=norm_filters, unique_filters=unique_filters)
        data = np.dot(spectrogram, filterbank)
        obj = np.asarray(data).view(cls)
        obj.spectrogram = spectrogram
        obj.filterbank = filterbank
        obj.crossover_frequencies = crossover_frequencies
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.filterbank = getattr(obj, 'filterbank', None)
        self.crossover_frequencies = getattr(obj, 'crossover_frequencies', None)