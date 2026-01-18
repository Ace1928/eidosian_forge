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
class NoteFormatter(mplticker.Formatter):
    """Ticker formatter for Notes

    Parameters
    ----------
    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    key : str
        Key for determining pitch spelling.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    LogHzFormatter
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

    def __init__(self, octave: bool=True, major: bool=True, key: str='C:maj', unicode: bool=True):
        self.octave = octave
        self.major = major
        self.key = key
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Apply the formatter to position"""
        if x <= 0:
            return ''
        vmin, vmax = self.axis.get_view_interval()
        if not self.major and vmax > 4 * max(1, vmin):
            return ''
        cents = vmax <= 2 * max(1, vmin)
        return core.hz_to_note(x, octave=self.octave, cents=cents, key=self.key, unicode=self.unicode)