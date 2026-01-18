from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def _parse_k_points(self) -> None:
    """Parse the list of k-points used in the calculation."""
    n_kpts = self.parse_scalar('n_kpts')
    if n_kpts is None:
        self._cache.update({'k_points': None, 'k_point_weights': None})
        return
    n_kpts = int(n_kpts)
    line_start = self.reverse_search_for(['| K-points in task'])
    line_end = self.reverse_search_for(['| k-point:'])
    if line_start == LINE_NOT_FOUND or line_end == LINE_NOT_FOUND or line_end - line_start != n_kpts:
        self._cache.update({'k_points': None, 'k_point_weights': None})
        return
    k_points = np.zeros((n_kpts, 3))
    k_point_weights = np.zeros(n_kpts)
    for kk, line in enumerate(self.lines[line_start + 1:line_end + 1]):
        k_points[kk] = [float(inp) for inp in line.split()[4:7]]
        k_point_weights[kk] = float(line.split()[-1])
    self._cache.update({'k_points': k_points, 'k_point_weights': k_point_weights})