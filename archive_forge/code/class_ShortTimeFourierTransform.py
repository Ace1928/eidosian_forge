from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import scipy.fftpack as fftpack
from ..processors import Processor
from .signal import Signal, FramedSignal
class ShortTimeFourierTransform(_PropertyMixin, np.ndarray):
    """
    ShortTimeFourierTransform class.

    Parameters
    ----------
    frames : :class:`.audio.signal.FramedSignal` instance
        Framed signal.
    window : numpy ufunc or numpy array, optional
        Window (function); if a function (e.g. `np.hanning`) is given, a window
        with the frame size of `frames` and the given shape is created.
    fft_size : int, optional
        FFT size (should be a power of 2); if 'None', the `frame_size` given by
        `frames` is used, if the given `fft_size` is greater than the
        `frame_size`, the frames are zero-padded accordingly.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.
    include_nyquist : bool, optional
        Include the Nyquist frequency bin (sample rate / 2).
    fftw : :class:`pyfftw.FFTW` instance, optional
        If a :class:`pyfftw.FFTW` object is given it is used to compute the
        STFT with the FFTW library. If 'None', a new :class:`pyfftw.FFTW`
        object is built. Requires 'pyfftw'.
    kwargs : dict, optional
        If no :class:`.audio.signal.FramedSignal` instance was given, one is
        instantiated with these additional keyword arguments.

    Notes
    -----
    If the :class:`Signal` (wrapped in the :class:`FramedSignal`) has an
    integer dtype, the `window` is automatically scaled as if the `signal` had
    a float dtype with the values being in the range [-1, 1]. This results in
    same valued STFTs independently of the dtype of the signal. On the other
    hand, this prevents extra memory consumption since the data-type of the
    signal does not need to be converted (and if no decoding is needed, the
    audio signal can be memory-mapped).

    Examples
    --------
    Create a :class:`ShortTimeFourierTransform` from a :class:`Signal` or
    :class:`FramedSignal`:

    >>> sig = Signal('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2494, -2510, ...,   655,   639], dtype=int16)
    >>> frames = FramedSignal(sig, frame_size=2048, hop_size=441)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> stft = ShortTimeFourierTransform(frames)
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[-3.15249+0.j     ,  2.62216-3.02425j, ...,
                                -0.03634-0.00005j,  0.0367 +0.00029j],
                               [-4.28429+0.j     ,  2.02009+2.01264j, ...,
                                -0.01981-0.00933j, -0.00536+0.02162j],
                               ...,
                               [-4.92274+0.j     ,  4.09839-9.42525j, ...,
                                 0.0055 -0.00257j,  0.00137+0.00577j],
                               [-9.22709+0.j     ,  8.76929+4.0005j , ...,
                                 0.00981-0.00014j, -0.00984+0.00006j]],
                              dtype=complex64)

    A ShortTimeFourierTransform can be instantiated directly from a file name:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav')
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[...]], dtype=complex64)

    Doing the same with a Signal of float data-type will result in a STFT of
    same value range (rounding errors will occur of course):

    >>> sig = Signal('tests/data/audio/sample.wav', dtype=np.float)
    >>> sig  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Signal([-0.07611, -0.0766 , ...,  0.01999,  0.0195 ])
    >>> frames = FramedSignal(sig, frame_size=2048, hop_size=441)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> stft = ShortTimeFourierTransform(frames)
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[-3.1524 +0.j     ,  2.62208-3.02415j, ...,
                                -0.03633-0.00005j,  0.0367 +0.00029j],
                               [-4.28416+0.j     ,  2.02003+2.01257j, ...,
                                -0.01981-0.00933j, -0.00536+0.02162j],
                               ...,
                               [-4.92259+0.j     ,  4.09827-9.42496j, ...,
                                 0.0055 -0.00257j,  0.00137+0.00577j],
                               [-9.22681+0.j     ,  8.76902+4.00038j, ...,
                                 0.00981-0.00014j, -0.00984+0.00006j]],
                              dtype=complex64)

    Additional arguments are passed to :class:`FramedSignal` and
    :class:`Signal` respectively:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav', frame_size=2048, fps=100, sample_rate=22050)
    >>> stft.frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> stft.frames.frame_size
    2048
    >>> stft.frames.hop_size
    220.5
    >>> stft.frames.signal.sample_rate
    22050

    """

    def __init__(self, frames, window=np.hanning, fft_size=None, circular_shift=False, include_nyquist=False, fft_window=None, fftw=None, **kwargs):
        pass

    def __new__(cls, frames, window=np.hanning, fft_size=None, circular_shift=False, include_nyquist=False, fft_window=None, fftw=None, **kwargs):
        if isinstance(frames, ShortTimeFourierTransform):
            frames = frames.frames
        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)
        frame_size = frames.shape[1]
        if fft_window is None:
            if hasattr(window, '__call__'):
                window = window(frame_size)
            try:
                max_range = float(np.iinfo(frames.signal.dtype).max)
                try:
                    fft_window = window / max_range
                except TypeError:
                    fft_window = np.ones(frame_size) / max_range
            except ValueError:
                fft_window = window
        try:
            fftw = rfft_builder(fft_window, fft_size, axis=0)
        except AttributeError:
            pass
        data = stft(frames, fft_window, fft_size=fft_size, circular_shift=circular_shift, include_nyquist=include_nyquist, fftw=fftw)
        obj = np.asarray(data).view(cls)
        obj.frames = frames
        obj.window = window
        obj.fft_window = fft_window
        obj.fft_size = fft_size if fft_size else frame_size
        obj.circular_shift = circular_shift
        obj.include_nyquist = include_nyquist
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.frames = getattr(obj, 'frames', None)
        self.window = getattr(obj, 'window', np.hanning)
        self.fft_window = getattr(obj, 'fft_window', None)
        self.fftw = getattr(obj, 'fftw', None)
        self.fft_size = getattr(obj, 'fft_size', None)
        self.circular_shift = getattr(obj, 'circular_shift', False)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return fft_frequencies(self.num_bins, self.frames.signal.sample_rate)

    def spec(self, **kwargs):
        """
        Returns the magnitude spectrogram of the STFT.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to
            :class:`.audio.spectrogram.Spectrogram`.

        Returns
        -------
        spec : :class:`.audio.spectrogram.Spectrogram`
            :class:`.audio.spectrogram.Spectrogram` instance.

        """
        from .spectrogram import Spectrogram
        return Spectrogram(self, **kwargs)

    def phase(self, **kwargs):
        """
        Returns the phase of the STFT.

        Parameters
        ----------
        kwargs : dict, optional
            keyword arguments passed to :class:`Phase`.

        Returns
        -------
        phase : :class:`Phase`
            :class:`Phase` instance.

        """
        return Phase(self, **kwargs)