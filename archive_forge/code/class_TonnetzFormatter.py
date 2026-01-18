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
class TonnetzFormatter(mplticker.Formatter):
    """A formatter for tonnetz axes

    See Also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(6)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.TonnetzFormatter())
    >>> ax.set(ylabel='Tonnetz')
    """

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Format for tonnetz positions"""
        return ['5$_x$', '5$_y$', 'm3$_x$', 'm3$_y$', 'M3$_x$', 'M3$_y$'][int(x)]