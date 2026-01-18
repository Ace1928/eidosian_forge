from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def _parse_hirshfeld(self) -> None:
    """Parse the Hirshfled charges volumes, and dipole moments."""
    line_start = self.reverse_search_for(['Performing Hirshfeld analysis of fragment charges and moments.'])
    if line_start == LINE_NOT_FOUND:
        self._cache.update({'hirshfeld_charges': None, 'hirshfeld_volumes': None, 'hirshfeld_atomic_dipoles': None, 'hirshfeld_dipole': None})
        return
    line_inds = self.search_for_all('Hirshfeld charge', line_start, -1)
    hirshfeld_charges = np.array([float(self.lines[ind].split(':')[1]) for ind in line_inds])
    line_inds = self.search_for_all('Hirshfeld volume', line_start, -1)
    hirshfeld_volumes = np.array([float(self.lines[ind].split(':')[1]) for ind in line_inds])
    line_inds = self.search_for_all('Hirshfeld dipole vector', line_start, -1)
    hirshfeld_atomic_dipoles = np.array([[float(inp) for inp in self.lines[ind].split(':')[1].split()] for ind in line_inds])
    if self.lattice is None:
        hirshfeld_dipole = np.sum(hirshfeld_charges.reshape((-1, 1)) * self.coords, axis=1)
    else:
        hirshfeld_dipole = None
    self._cache.update({'hirshfeld_charges': hirshfeld_charges, 'hirshfeld_volumes': hirshfeld_volumes, 'hirshfeld_atomic_dipoles': hirshfeld_atomic_dipoles, 'hirshfeld_dipole': hirshfeld_dipole})