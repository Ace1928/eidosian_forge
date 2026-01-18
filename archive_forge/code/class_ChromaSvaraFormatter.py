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
class ChromaSvaraFormatter(mplticker.Formatter):
    """A formatter for chroma axes with svara instead of notes.

    If mela is given, Carnatic svara names will be used.

    Otherwise, Hindustani svara names will be used.

    If `Sa` is not given, it will default to 0 (equivalent to `C`).

    See Also
    --------
    ChromaFormatter

    """

    def __init__(self, Sa: Optional[float]=None, mela: Optional[Union[int, str]]=None, abbr: bool=True, unicode: bool=True):
        if Sa is None:
            Sa = 0
        self.Sa = Sa
        self.mela = mela
        self.abbr = abbr
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Format for chroma positions"""
        if self.mela is not None:
            return core.midi_to_svara_c(int(x), Sa=self.Sa, mela=self.mela, octave=False, abbr=self.abbr, unicode=self.unicode)
        else:
            return core.midi_to_svara_h(int(x), Sa=self.Sa, octave=False, abbr=self.abbr, unicode=self.unicode)