from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class MultiBandSpectrogramProcessor(Processor):
    """
    Spectrogram processor which combines the spectrogram magnitudes into
    multiple bands.

    Parameters
    ----------
    crossover_frequencies : list or numpy array
        List of crossover frequencies at which a spectrogram is split into
        the individual bands.
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

    """

    def __init__(self, crossover_frequencies, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        self.crossover_frequencies = np.array(crossover_frequencies)
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters

    def process(self, data, **kwargs):
        """
        Return the a multi-band representation of the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`MultiBandSpectrogram`.

        Returns
        -------
        multi_band_spec : :class:`MultiBandSpectrogram` instance
            Spectrogram split into multiple bands.

        """
        args = dict(crossover_frequencies=self.crossover_frequencies, fmin=self.fmin, fmax=self.fmax, norm_filters=self.norm_filters, unique_filters=self.unique_filters)
        args.update(kwargs)
        return MultiBandSpectrogram(data, **args)