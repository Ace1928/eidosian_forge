from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
class Kpoint(MSONable):
    """Class to store kpoint objects. A kpoint is defined with a lattice and frac
    or Cartesian coordinates syntax similar than the site object in
    pymatgen.core.structure.
    """

    def __init__(self, coords: np.ndarray, lattice: Lattice, to_unit_cell: bool=False, coords_are_cartesian: bool=False, label: str | None=None) -> None:
        """
        Args:
            coords: coordinate of the kpoint as a numpy array
            lattice: A pymatgen.core.Lattice object representing
                the reciprocal lattice of the kpoint
            to_unit_cell: Translates fractional coordinate to the basic unit
                cell, i.e., all fractional coordinates satisfy 0 <= a < 1.
                Defaults to False.
            coords_are_cartesian: Boolean indicating if the coordinates given are
                in Cartesian or fractional coordinates (by default fractional)
            label: the label of the kpoint if any (None by default).
        """
        self._lattice = lattice
        self._frac_coords = lattice.get_fractional_coords(coords) if coords_are_cartesian else coords
        self._label = label
        if to_unit_cell:
            for idx, fc in enumerate(self._frac_coords):
                self._frac_coords[idx] -= math.floor(fc)
        self._cart_coords = lattice.get_cartesian_coords(self._frac_coords)

    @property
    def lattice(self) -> Lattice:
        """The lattice associated with the kpoint. It's a
        pymatgen.core.Lattice object.
        """
        return self._lattice

    @property
    def label(self) -> str | None:
        """The label associated with the kpoint."""
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        """Set the label of the kpoint."""
        self._label = label

    @property
    def frac_coords(self) -> np.ndarray:
        """The fractional coordinates of the kpoint as a numpy array."""
        return np.copy(self._frac_coords)

    @property
    def cart_coords(self) -> np.ndarray:
        """The Cartesian coordinates of the kpoint as a numpy array."""
        return np.copy(self._cart_coords)

    @property
    def a(self) -> float:
        """Fractional a coordinate of the kpoint."""
        return self._frac_coords[0]

    @property
    def b(self) -> float:
        """Fractional b coordinate of the kpoint."""
        return self._frac_coords[1]

    @property
    def c(self) -> float:
        """Fractional c coordinate of the kpoint."""
        return self._frac_coords[2]

    def __str__(self) -> str:
        """Returns a string with fractional, Cartesian coordinates and label."""
        return f'{self.frac_coords} {self.cart_coords} {self.label}'

    def __eq__(self, other: object) -> bool:
        """Check if two kpoints are equal."""
        if not isinstance(other, Kpoint):
            return NotImplemented
        return np.allclose(self.frac_coords, other.frac_coords) and self.lattice == other.lattice and (self.label == other.label)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation of a kpoint."""
        return {'lattice': self.lattice.as_dict(), 'fcoords': self.frac_coords.tolist(), 'ccoords': self.cart_coords.tolist(), 'label': self.label, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Create from dict.

        Args:
            dct (dict): A dict with all data for a kpoint object.

        Returns:
            Kpoint
        """
        lattice = Lattice.from_dict(dct['lattice'])
        return cls(coords=dct['fcoords'], lattice=lattice, coords_are_cartesian=False, label=dct['label'])