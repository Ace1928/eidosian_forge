from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def _parse_initial_charges_and_moments(self) -> None:
    """Parse the initial charges and magnetic moments from a file"""
    charges = np.zeros(self.n_atoms)
    magmoms = None
    line_start = self.reverse_search_for(['Initial charges', 'Initial moments and charges'])
    if line_start != LINE_NOT_FOUND:
        line_start += 2
        magmoms = np.zeros(self.n_atoms)
        for ll, line in enumerate(self.lines[line_start:line_start + self.n_atoms]):
            inp = line.split()
            if len(inp) == 4:
                charges[ll] = float(inp[2])
                magmoms = None
            else:
                charges[ll] = float(inp[3])
                magmoms[ll] = float(inp[2])
    self._cache['initial_charges'] = charges
    self._cache['initial_magnetic_moments'] = magmoms