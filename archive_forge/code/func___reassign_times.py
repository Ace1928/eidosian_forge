import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def __reassign_times(y: np.ndarray, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, np.ndarray]:
    """Time reassignments based on a spectrogram representation.

    The reassignment vector is calculated using equation 5.23 in Flandrin,
    Auger, & Chassande-Mottin 2002::

        t_reassigned = t + np.real(S_th/S_h)

    where ``S_h`` is the complex STFT calculated using the original window, and
    ``S_th`` is the complex STFT calculated using the original window multiplied
    by the time offset from the window center.

    See `reassigned_spectrogram` for references.

    It is recommended to use ``pad_mode="constant"`` (zero padding) or else
    ``center=False``, rather than the defaults. Time reassignment assumes that
    the energy in each FFT bin is associated with exactly one impulse event.
    Reflection padding at the edges of the signal may invalidate the reassigned
    estimates in the boundary frames.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to `__reassign_times`

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        See `stft` for details.

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    dtype : numeric type
        Complex numeric type for ``S``. Default is inferred to match
        the precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    times : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Reassigned times:
        ``times[f, t]`` is the time for bin ``f``, frame ``t``.
    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    Warns
    -----
    RuntimeWarning
        Time estimates with zero support will produce a divide-by-zero warning
        and will be returned as `np.nan`.

    See Also
    --------
    stft : Short-time Fourier Transform
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> times, S = librosa.core.spectrum.__reassign_times(y, sr=sr)
    >>> times
    array([[ 2.268e-05,  1.144e-02, ...,  5.332e+00,  5.333e+00],
           [ 2.268e-05,  1.451e-02, ...,  5.334e+00,  5.333e+00],
           ...,
           [ 2.268e-05, -6.177e-04, ...,  5.368e+00,  5.327e+00],
           [ 2.268e-05,  1.420e-03, ...,  5.307e+00,  5.328e+00]])
    """
    if win_length is None:
        win_length = n_fft
    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)
    if hop_length is None:
        hop_length = int(win_length // 4)
    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)
        S_h = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    else:
        if dtype is None:
            dtype = S.dtype
        S_h = S
    half_width = n_fft // 2
    window_times: np.ndarray
    if n_fft % 2:
        window_times = np.arange(-half_width, half_width + 1)
    else:
        window_times = np.arange(0.5 - half_width, half_width)
    window_time_weighted = window * window_times
    S_th = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window_time_weighted, center=center, dtype=dtype, pad_mode=pad_mode)
    correction = np.real(S_th / S_h)
    if center:
        pad_length = None
    else:
        pad_length = n_fft
    times = convert.frames_to_time(np.arange(S_h.shape[-1]), sr=sr, hop_length=hop_length, n_fft=pad_length)
    times = util.expand_to(times, ndim=correction.ndim, axes=-1) + correction / sr
    return (times, S_h)