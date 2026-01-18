from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class LogarithmicFilteredSpectrogram(LogarithmicSpectrogram, FilteredSpectrogram):
    """
    LogarithmicFilteredSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`FilteredSpectrogram` instance
        Filtered spectrogram.
    kwargs : dict, optional
        If no :class:`FilteredSpectrogram` instance was given, one is
        instantiated with these additional keyword arguments and
        logarithmically scaled afterwards, i.e. passed to
        :class:`LogarithmicSpectrogram`.

    Notes
    -----
    For the filtering and scaling parameters, please refer to
    :class:`FilteredSpectrogram` and :class:`LogarithmicSpectrogram`.

    See Also
    --------
    :class:`FilteredSpectrogram`
    :class:`LogarithmicSpectrogram`

    Examples
    --------
    Create a :class:`LogarithmicFilteredSpectrogram` from a
    :class:`Spectrogram` (or anything it can be instantiated from. This is
    mainly a convenience class which first filters the spectrogram and then
    scales it logarithmically.

    >>> spec = LogarithmicFilteredSpectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicFilteredSpectrogram([[0.82358, 0.86341, ..., 0.02295, 0.02719],
                                    [0.97509, 0.98658, ..., 0.03223, 0.0375 ],
                                    ...,
                                    [1.04322, 0.32637, ..., 0.02065, 0.01821],
                                    [0.98236, 0.89276, ..., 0.01587, 0.0144 ]],
                                    dtype=float32)
    >>> spec.shape
    (281, 81)
    >>> spec.filterbank  # doctest: +ELLIPSIS
    LogarithmicFilterbank([[...]], dtype=float32)
    >>> spec.min()  # doctest: +ELLIPSIS
    LogarithmicFilteredSpectrogram(0.00831, dtype=float32)

    """

    def __init__(self, spectrogram, **kwargs):
        pass

    def __new__(cls, spectrogram, **kwargs):
        mul = kwargs.pop('mul', MUL)
        add = kwargs.pop('add', ADD)
        if not isinstance(spectrogram, FilteredSpectrogram):
            spectrogram = FilteredSpectrogram(spectrogram, **kwargs)
        data = LogarithmicSpectrogram(spectrogram, mul=mul, add=add, **kwargs)
        obj = np.asarray(data).view(cls)
        obj.mul = data.mul
        obj.add = data.add
        obj.stft = spectrogram.stft
        obj.spectrogram = spectrogram
        return obj

    @property
    def filterbank(self):
        """Filterbank."""
        return self.spectrogram.filterbank

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.filterbank.center_frequencies