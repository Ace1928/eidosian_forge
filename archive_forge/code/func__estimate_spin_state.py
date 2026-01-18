from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def _estimate_spin_state(species: str | Species, motif: Literal['oct', 'tet'], known_magmom: float) -> Literal['undefined', 'low', 'high', 'unknown']:
    """Simple heuristic to estimate spin state. If magnetic moment
        is sufficiently close to that predicted for a given spin state,
        we assign it that state. If we only have data for one spin
        state then that's the one we use (e.g. we assume all tetrahedral
        complexes are high-spin, since this is typically the case).

        Args:
            species: str or Species
            motif ("oct" | "tet"): Tetrahedron or octahedron crystal site coordination
            known_magmom: magnetic moment in Bohr magnetons

        Returns:
            "undefined" (if only one spin state possible), "low", "high" or "unknown"
        """
    mu_so_high = JahnTellerAnalyzer.mu_so(species, motif=motif, spin_state='high')
    mu_so_low = JahnTellerAnalyzer.mu_so(species, motif=motif, spin_state='low')
    if mu_so_high == mu_so_low:
        return 'undefined'
    if mu_so_high is None:
        return 'low'
    if mu_so_low is None:
        return 'high'
    diff = mu_so_high - mu_so_low
    if known_magmom > mu_so_high or abs(mu_so_high - known_magmom) < diff * 0.25:
        return 'high'
    if known_magmom < mu_so_low or abs(mu_so_low - known_magmom) < diff * 0.25:
        return 'low'
    return 'unknown'