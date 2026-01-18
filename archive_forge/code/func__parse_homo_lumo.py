from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def _parse_homo_lumo(self) -> dict[str, float]:
    """Parse the HOMO/LUMO values and get band gap if periodic."""
    line_start = self.reverse_search_for(['Highest occupied state (VBM)'])
    homo = float(self.lines[line_start].split(' at ')[1].split('eV')[0].strip())
    line_start = self.reverse_search_for(['Lowest unoccupied state (CBM)'])
    lumo = float(self.lines[line_start].split(' at ')[1].split('eV')[0].strip())
    line_start = self.reverse_search_for(['verall HOMO-LUMO gap'])
    homo_lumo_gap = float(self.lines[line_start].split(':')[1].split('eV')[0].strip())
    line_start = self.reverse_search_for(['Smallest direct gap'])
    if line_start == LINE_NOT_FOUND:
        return {'vbm': homo, 'cbm': lumo, 'gap': homo_lumo_gap, 'direct_gap': homo_lumo_gap}
    direct_gap = float(self.lines[line_start].split(':')[1].split('eV')[0].strip())
    return {'vbm': homo, 'cbm': lumo, 'gap': homo_lumo_gap, 'direct_gap': direct_gap}