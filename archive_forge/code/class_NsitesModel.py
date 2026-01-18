from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class NsitesModel(EnergyModel):
    """
    Sets the energy to the number of sites. More sites => higher "energy".
    Used to rank structures from smallest number of sites to largest number
    of sites after enumeration.
    """

    def get_energy(self, structure: Structure):
        """
        Args:
            structure: Structure

        Returns:
            Energy value
        """
        return len(structure)

    def as_dict(self):
        """MSONable dict"""
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'init_args': {}}