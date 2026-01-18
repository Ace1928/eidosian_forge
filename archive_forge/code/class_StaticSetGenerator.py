from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from pymatgen.core import Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
@dataclass
class StaticSetGenerator(AimsInputGenerator):
    """Common class for ground-state generators.

    Parameters
    ----------
    calc_type: str
        The type of calculation
    """
    calc_type: str = 'static'

    def get_parameter_updates(self, structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict[str, Any]:
        """Get the parameter updates for the calculation.

        Parameters
        ----------
        structure: Structure or Molecule
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns:
            dict: The updated for the parameters for the output section of FHI-aims
        """
        return prev_parameters