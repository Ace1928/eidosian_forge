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
class SvaraFormatter(mplticker.Formatter):
    """Ticker formatter for Svara

    Parameters
    ----------
    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    Sa : number > 0
        Frequency (in Hz) of Sa

    mela : str or int
        For Carnatic svara, the index or name of the melakarta raga in question

        To use Hindustani svara, set ``mela=None``

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter
    librosa.hz_to_svara_c
    librosa.hz_to_svara_h


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.SvaraFormatter(261))
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, Sa: float, octave: bool=True, major: bool=True, abbr: bool=False, mela: Optional[Union[str, int]]=None, unicode: bool=True):
        if Sa is None:
            raise ParameterError('Sa frequency is required for svara display formatting')
        self.Sa = Sa
        self.octave = octave
        self.major = major
        self.abbr = abbr
        self.mela = mela
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        if x <= 0:
            return ''
        vmin, vmax = self.axis.get_view_interval()
        if not self.major and vmax > 4 * max(1, vmin):
            return ''
        if self.mela is None:
            return core.hz_to_svara_h(x, Sa=self.Sa, octave=self.octave, abbr=self.abbr, unicode=self.unicode)
        else:
            return core.hz_to_svara_c(x, Sa=self.Sa, mela=self.mela, octave=self.octave, abbr=self.abbr, unicode=self.unicode)