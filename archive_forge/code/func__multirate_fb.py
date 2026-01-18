import warnings
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from numba import jit
from ._cache import cache
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecated
from .core.convert import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from .core.convert import fft_frequencies, mel_frequencies
from numpy.typing import ArrayLike, DTypeLike
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
@cache(level=10)
def _multirate_fb(center_freqs: Optional[np.ndarray]=None, sample_rates: Optional[np.ndarray]=None, Q: float=25.0, passband_ripple: float=1, stopband_attenuation: float=50, ftype: str='ellip', flayout: str='sos') -> Tuple[List[Any], np.ndarray]:
    """Construct a multirate filterbank.

     A filter bank consists of multiple band-pass filters which divide the input signal
     into subbands. In the case of a multirate filter bank, the band-pass filters
     operate with resampled versions of the input signal, e.g. to keep the length
     of a filter constant while shifting its center frequency.

     This implementation uses `scipy.signal.iirdesign` to design the filters.

    Parameters
    ----------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.

    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Samplerate for each filter (used for multirate filterbank).

    Q : float
        Q factor (influences the filter bandwidth).

    passband_ripple : float
        The maximum loss in the passband (dB)
        See `scipy.signal.iirdesign` for details.

    stopband_attenuation : float
        The minimum attenuation in the stopband (dB)
        See `scipy.signal.iirdesign` for details.

    ftype : str
        The type of IIR filter to design
        See `scipy.signal.iirdesign` for details.

    flayout : string
        Valid `output` argument for `scipy.signal.iirdesign`.

        - If `ba`, returns numerators/denominators of the transfer functions,
          used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.

        - If `sos`, returns a series of second-order filters,
          used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.

        - If `zpk`, returns zeros, poles, and system gains of the transfer functions.

    Returns
    -------
    filterbank : list [shape=(n,), dtype=float]
        Each list entry comprises the filter coefficients for a single filter.
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Samplerate for each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    scipy.signal.iirdesign

    Raises
    ------
    ParameterError
        If ``center_freqs`` is ``None``.
        If ``sample_rates`` is ``None``.
        If ``center_freqs.shape`` does not match ``sample_rates.shape``.
    """
    if center_freqs is None:
        raise ParameterError('center_freqs must be provided.')
    if sample_rates is None:
        raise ParameterError('sample_rates must be provided.')
    if center_freqs.shape != sample_rates.shape:
        raise ParameterError('Number of provided center_freqs and sample_rates must be equal.')
    nyquist = 0.5 * sample_rates
    filter_bandwidths = center_freqs / float(Q)
    filterbank = []
    for cur_center_freq, cur_nyquist, cur_bw in zip(center_freqs, nyquist, filter_bandwidths):
        passband_freqs = [cur_center_freq - 0.5 * cur_bw, cur_center_freq + 0.5 * cur_bw] / cur_nyquist
        stopband_freqs = [cur_center_freq - cur_bw, cur_center_freq + cur_bw] / cur_nyquist
        cur_filter = scipy.signal.iirdesign(passband_freqs, stopband_freqs, passband_ripple, stopband_attenuation, analog=False, ftype=ftype, output=flayout)
        filterbank.append(cur_filter)
    return (filterbank, sample_rates)