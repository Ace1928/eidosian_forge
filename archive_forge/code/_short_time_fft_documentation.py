from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
Return minimum and maximum values time-frequency values.

        A tuple with four floats  ``(t0, t1, f0, f1)`` for 'tf' and
        ``(f0, f1, t0, t1)`` for 'ft') is returned describing the corners
        of the time-frequency domain of the `~ShortTimeFFT.stft`.
        That tuple can be passed to `matplotlib.pyplot.imshow` as a parameter
        with the same name.

        Parameters
        ----------
        n : int
            Number of samples in input signal.
        axes_seq : {'tf', 'ft'}
            Return time extent first and then frequency extent or vice-versa.
        center_bins: bool
            If set (default ``False``), the values of the time slots and
            frequency bins are moved from the side the middle. This is useful,
            when plotting the `~ShortTimeFFT.stft` values as step functions,
            i.e., with no interpolation.

        See Also
        --------
        :func:`matplotlib.pyplot.imshow`: Display data as an image.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        