from __future__ import annotations
import linecache
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.kpath import KPathSeek
class ACExtractorBase(ABC):
    """A parent class of ACExtractor and ACstrExtractor, ensuring that they are as consistent as possible"""

    @abstractmethod
    def get_n_atoms(self) -> int:
        """Get the number of atoms in structure defined by atom.config file."""

    @abstractmethod
    def get_lattice(self) -> np.ndarray:
        """Get the lattice of structure defined by atom.config file"""

    @abstractmethod
    def get_types(self) -> np.ndarray:
        """Get atomic number of atoms in structure defined by atom.config file"""

    @abstractmethod
    def get_coords(self) -> np.ndarray:
        """Get fractional coordinates of atoms in structure defined by atom.config file"""

    @abstractmethod
    def get_magmoms(self) -> np.ndarray:
        """Get atomic magmoms of atoms in structure defined by atom.config file"""