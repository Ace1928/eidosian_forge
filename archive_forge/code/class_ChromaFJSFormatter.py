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
class ChromaFJSFormatter(mplticker.Formatter):
    """A formatter for chroma axes with functional just notation

    See Also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(12)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.ChromaFJSFormatter(intervals="ji5", bins_per_octave=12))
    >>> ax.set(ylabel='Pitch class')
    """

    def __init__(self, *, intervals: Union[str, Collection[float]], unison: str='C', unicode: bool=True, bins_per_octave: Optional[int]=None):
        self.unison = unison
        self.unicode = unicode
        self.intervals = intervals
        try:
            if not isinstance(intervals, str):
                bins_per_octave = len(intervals)
            if not isinstance(bins_per_octave, int):
                raise ParameterError(f'bins_per_octave={bins_per_octave} must be integer-valued')
            self.bins_per_octave: int = bins_per_octave
            self.intervals_ = core.interval_frequencies(self.bins_per_octave, fmin=1, intervals=intervals, bins_per_octave=self.bins_per_octave)
        except TypeError as exc:
            raise ParameterError(f'intervals={intervals} must be of type str or a collection of numbers between 1 and 2') from exc

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Format for chroma positions"""
        lab: str = core.interval_to_fjs(self.intervals_[int(x) % self.bins_per_octave], unison=self.unison, unicode=self.unicode)
        return lab