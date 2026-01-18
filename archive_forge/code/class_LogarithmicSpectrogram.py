from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class LogarithmicSpectrogram(Spectrogram):
    """
    LogarithmicSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    log : numpy ufunc, optional
        Logarithmic scaling function to apply.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking
        the logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated
        with these additional keyword arguments.

    Examples
    --------
    Create a :class:`LogarithmicSpectrogram` from a :class:`Spectrogram` (or
    anything it can be instantiated from. Per default `np.log10` is used as
    the scaling function and a value of 1 is added to avoid negative values.

    >>> spec = LogarithmicSpectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicSpectrogram([[...]], dtype=float32)
    >>> spec.min()
    LogarithmicSpectrogram(0., dtype=float32)

    """

    def __init__(self, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        pass

    def __new__(cls, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
            data = spectrogram
        else:
            data = spectrogram.copy()
        if mul is not None:
            data *= mul
        if add is not None:
            data += add
        if log is not None:
            log(data, data)
        obj = np.asarray(data).view(cls)
        obj.mul = mul
        obj.add = add
        obj.stft = spectrogram.stft
        obj.spectrogram = spectrogram
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stft = getattr(obj, 'stft', None)
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.mul = getattr(obj, 'mul', MUL)
        self.add = getattr(obj, 'add', ADD)

    @property
    def filterbank(self):
        """Filterbank."""
        return self.spectrogram.filterbank

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.spectrogram.bin_frequencies