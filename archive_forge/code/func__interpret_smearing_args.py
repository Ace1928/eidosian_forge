from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
@staticmethod
def _interpret_smearing_args(npts: int, width: float=None, default_npts: int=1000, default_width: float=0.1) -> Tuple[int, Union[float, None]]:
    """Figure out what the user intended: resample if width provided"""
    if width is not None:
        if npts:
            return (npts, float(width))
        else:
            return (default_npts, float(width))
    elif npts:
        return (npts, default_width)
    else:
        return (0, None)