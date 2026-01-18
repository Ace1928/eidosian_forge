from __future__ import annotations
from itertools import product
import warnings
import numpy as np
import matplotlib.cm as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt
from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co
class FJSFormatter(mplticker.Formatter):
    """Ticker formatter for Functional Just System (FJS) notation

    Parameters
    ----------
    fmin : float
        The unison frequency for this axis

    intervals : str or array of float in [1, 2)
        The interval specification for the frequency axis.

        See `core.interval_frequencies` for supported values.

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    unison : str
        The unison note name.  If not provided, it will be inferred from fmin.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    NoteFormatter
    hz_to_fjs
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, *, fmin: int, n_bins: int, bins_per_octave: int, intervals: Union[str, Collection[float]], major: bool=True, unison: Optional[str]=None, unicode: bool=True):
        self.fmin = fmin
        self.major = major
        self.unison = unison
        self.unicode = unicode
        self.intervals = intervals
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.frequencies_ = core.interval_frequencies(n_bins, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave)

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Apply the formatter to position"""
        if x <= 0:
            return ''
        vmin, vmax = self.axis.get_view_interval()
        if not self.major and vmax > 4 * max(1, vmin):
            return ''
        idx = util.match_events(np.atleast_1d(x), self.frequencies_)[0]
        label: str = core.hz_to_fjs(self.frequencies_[idx], fmin=self.fmin, unison=self.unison, unicode=self.unicode)
        return label