from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class EnergyModel(MSONable, abc.ABC):
    """Abstract structure filter class."""

    @abc.abstractmethod
    def get_energy(self, structure) -> float:
        """
        Args:
            structure: Structure

        Returns:
            Energy value
        """
        return 0.0

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            EnergyModel
        """
        return cls(**dct['init_args'])