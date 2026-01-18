from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath
@dataclass
class BandStructureSetGenerator(AimsInputGenerator):
    """A generator for the band structure calculation input set.

    Parameters
    ----------
    calc_type: str
        The type of calculations
    k_point_density: float
        The number of k_points per angstrom
    """
    calc_type: str = 'bands'
    k_point_density: float = 20

    def get_parameter_updates(self, structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict[str, Sequence[str]]:
        """Get the parameter updates for the calculation.

        Parameters
        ----------
        structure: Structure
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns:
            dict: The updated for the parameters for the output section of FHI-aims
        """
        if isinstance(structure, Molecule):
            raise ValueError('BandStructures can not be made for non-periodic systems')
        updated_outputs = prev_parameters.get('output', [])
        updated_outputs += prepare_band_input(structure, self.k_point_density)
        return {'output': updated_outputs}