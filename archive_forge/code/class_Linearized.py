from typing import Tuple
import numpy as np
from ase.units import Bohr, Ha
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
class Linearized:

    def __init__(self):
        self._data = {'CC': (1.53, 1.69, 7.43, 0.71, 0.37), 'BN': (1.56, 1.58, 4.22, 0.42, 0.9)}

    def __call__(self, el1: str, el2: str, length: float) -> Tuple[float, float]:
        """Bond polarizability

        Parameters
        ----------
        el1: element string
        el2: element string
        length: float

        Returns
        -------
        alphal: float
          Parallel component
        alphap: float
          Perpendicular component
        """
        if el1 > el2:
            bond = el2 + el1
        else:
            bond = el1 + el2
        assert bond in self._data
        length0, al, ald, ap, apd = self._data[bond]
        return (al + ald * (length - length0), ap + apd * (length - length0))