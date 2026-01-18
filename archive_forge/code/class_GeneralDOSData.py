from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
class GeneralDOSData(DOSData):
    """Base class for a single series of DOS-like data

    Only the 'info' is a mutable attribute; DOS data is set at init

    This is the base class for DOSData objects that accept/set seperate
    "energies" and "weights" sequences of equal length at init.

    """

    def __init__(self, energies: Union[Sequence[float], np.ndarray], weights: Union[Sequence[float], np.ndarray], info: Info=None) -> None:
        super().__init__(info=info)
        n_entries = len(energies)
        if len(weights) != n_entries:
            raise ValueError('Energies and weights must be the same length')
        self._data = np.empty((2, n_entries), dtype=float, order='C')
        self._data[0, :] = energies
        self._data[1, :] = weights

    def get_energies(self) -> np.ndarray:
        return self._data[0, :].copy()

    def get_weights(self) -> np.ndarray:
        return self._data[1, :].copy()
    D = TypeVar('D', bound='GeneralDOSData')

    def copy(self: D) -> D:
        return type(self)(self.get_energies(), self.get_weights(), info=self.info.copy())