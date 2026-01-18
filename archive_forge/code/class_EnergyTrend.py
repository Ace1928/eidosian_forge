from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
class EnergyTrend:
    """Class for fitting trends to energies."""

    def __init__(self, energies):
        """
        Args:
            energies: Energies
        """
        self.energies = energies

    def spline(self):
        """Fit spline to energy trend data."""
        return UnivariateSpline(range(len(self.energies)), self.energies, k=4)

    def smoothness(self):
        """Get rms average difference between spline and energy trend."""
        energies = self.energies
        try:
            sp = self.spline()
        except Exception:
            print('Energy spline failed.')
            return None
        spline_energies = sp(range(len(energies)))
        diff = spline_energies - energies
        return np.sqrt(np.sum(np.square(diff)) / len(energies))

    def max_spline_jump(self):
        """Get maximum difference between spline and energy trend."""
        sp = self.spline()
        return max(self.energies - sp(range(len(self.energies))))

    def endpoints_minima(self, slope_cutoff=0.005):
        """Test if spline endpoints are at minima for a given slope cutoff."""
        energies = self.energies
        try:
            sp = self.spline()
        except Exception:
            print('Energy spline failed.')
            return None
        der = sp.derivative()
        der_energies = der(range(len(energies)))
        return {'polar': abs(der_energies[-1]) <= slope_cutoff, 'nonpolar': abs(der_energies[0]) <= slope_cutoff}