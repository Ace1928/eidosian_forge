import numpy as np
import scipy
import scipy.stats
from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .feature import tempo as _tempo
from .util.exceptions import ParameterError
from .util.decorators import moved
from typing import Any, Callable, Optional, Tuple
def __beat_tracker(onset_envelope: np.ndarray, bpm: float, fft_res: float, tightness: float, trim: bool) -> np.ndarray:
    """Tracks beats in an onset strength envelope.

    Parameters
    ----------
    onset_envelope : np.ndarray [shape=(n,)]
        onset strength envelope
    bpm : float [scalar]
        tempo estimate
    fft_res : float [scalar]
        resolution of the fft (sr / hop_length)
    tightness : float [scalar]
        how closely do we adhere to bpm?
    trim : bool [scalar]
        trim leading/trailing beats with weak onsets?

    Returns
    -------
    beats : np.ndarray [shape=(n,)]
        frame numbers of beat events
    """
    if bpm <= 0:
        raise ParameterError('bpm must be strictly positive')
    period = round(60.0 * fft_res / bpm)
    localscore = __beat_local_score(onset_envelope, period)
    backlink, cumscore = __beat_track_dp(localscore, period, tightness)
    beats = [__last_beat(cumscore)]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])
    beats = np.array(beats[::-1], dtype=int)
    beats = __trim_beats(localscore, beats, trim)
    return beats