from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
def _almost_equals(self, other: Any) -> bool:
    """Compare with another DOSData for testing purposes"""
    if not isinstance(other, type(self)):
        return False
    if self.info != other.info:
        return False
    if not np.allclose(self.get_weights(), other.get_weights()):
        return False
    return np.allclose(self.get_energies(), other.get_energies())