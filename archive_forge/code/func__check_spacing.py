from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
def _check_spacing(self, width) -> float:
    current_spacing = self._data[0, 1] - self._data[0, 0]
    if width < 2 * current_spacing:
        warnings.warn('The broadening width is small compared to the original sampling density. The results are unlikely to be smooth.')
    return current_spacing