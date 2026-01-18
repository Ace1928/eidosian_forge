from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from pymatgen.core import Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
@dataclass
class SocketIOSetGenerator(AimsInputGenerator):
    """Class to generate FHI-aims input sets for running with the socket.

    Parameters
    ----------
    calc_type: str
        The type of calculation
    host: str
        The hostname for the server the socket is on
    port: int
        The port the socket server is listening on
    """
    calc_type: str = 'multi_scf'
    host: str = 'localhost'
    port: int = 12345

    def get_parameter_updates(self, structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict:
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
        return {'use_pimd_wrapper': (self.host, self.port)}