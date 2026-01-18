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
def __coord_time(n: int, sr: float=22050, hop_length: int=512, **_kwargs: Any) -> np.ndarray:
    """Get time coordinates from frames"""
    times: np.ndarray = core.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return times