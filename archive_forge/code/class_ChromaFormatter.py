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
class ChromaFormatter(mplticker.Formatter):
    """A formatter for chroma axes

    See Also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(12)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
    >>> ax.set(ylabel='Pitch class')
    """

    def __init__(self, key: str='C:maj', unicode: bool=True):
        self.key = key
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Format for chroma positions"""
        return core.midi_to_note(int(x), octave=False, cents=False, key=self.key, unicode=self.unicode)