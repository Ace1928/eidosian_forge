from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class Spectrogram(np.ndarray):
    """
    A :class:`Spectrogram` represents the magnitude spectrogram of a
    :class:`.audio.stft.ShortTimeFourierTransform`.

    Parameters
    ----------
    stft : :class:`.audio.stft.ShortTimeFourierTransform` instance
        Short Time Fourier Transform.
    kwargs : dict, optional
        If no :class:`.audio.stft.ShortTimeFourierTransform` instance was
        given, one is instantiated with these additional keyword arguments.

    Examples
    --------
    Create a :class:`Spectrogram` from a
    :class:`.audio.stft.ShortTimeFourierTransform` (or anything it can be
    instantiated from:

    >>> spec = Spectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Spectrogram([[ 3.15249,  4.00272, ...,  0.03634,  0.03671],
                 [ 4.28429,  2.85158, ...,  0.0219 ,  0.02227],
                 ...,
                 [ 4.92274, 10.27775, ...,  0.00607,  0.00593],
                 [ 9.22709,  9.6387 , ...,  0.00981,  0.00984]], dtype=float32)

    """

    def __init__(self, stft, **kwargs):
        pass

    def __new__(cls, stft, **kwargs):
        from .stft import ShortTimeFourierTransform
        if isinstance(stft, Spectrogram):
            data = stft
        elif isinstance(stft, ShortTimeFourierTransform):
            data = np.abs(stft)
        else:
            stft = ShortTimeFourierTransform(stft, **kwargs)
            data = np.abs(stft)
        obj = np.asarray(data).view(cls)
        obj.stft = stft
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stft = getattr(obj, 'stft', None)

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.shape[1])

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies

    def diff(self, **kwargs):
        """
        Return the difference of the magnitude spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :class:`SpectrogramDifference`.

        Returns
        -------
        diff : :class:`SpectrogramDifference` instance
            The differences of the magnitude spectrogram.

        """
        return SpectrogramDifference(self, **kwargs)

    def filter(self, **kwargs):
        """
        Return a filtered version of the magnitude spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :class:`FilteredSpectrogram`.

        Returns
        -------
        filt_spec : :class:`FilteredSpectrogram` instance
            Filtered version of the magnitude spectrogram.

        """
        return FilteredSpectrogram(self, **kwargs)

    def log(self, **kwargs):
        """
        Return a logarithmically scaled version of the magnitude spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :class:`LogarithmicSpectrogram`.

        Returns
        -------
        log_spec : :class:`LogarithmicSpectrogram` instance
            Logarithmically scaled version of the magnitude spectrogram.

        """
        return LogarithmicSpectrogram(self, **kwargs)