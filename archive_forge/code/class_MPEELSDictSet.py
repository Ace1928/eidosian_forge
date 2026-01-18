from __future__ import annotations
import abc
import logging
import os
import sys
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.feff.inputs import Atoms, Header, Potential, Tags
class MPEELSDictSet(FEFFDictSet):
    """FeffDictSet for ELNES spectroscopy."""

    def __init__(self, absorbing_atom, structure, edge, spectrum, radius, beam_energy, beam_direction, collection_angle, convergence_angle, config_dict, user_eels_settings=None, nkpts: int=1000, user_tag_settings: dict | None=None, **kwargs):
        """
        Args:
            absorbing_atom (str/int): absorbing atom symbol or site index
            structure (Structure): input structure
            edge (str): absorption edge
            spectrum (str): ELNES or EXELFS
            radius (float): cluster radius in Angstroms.
            beam_energy (float): Incident beam energy in keV
            beam_direction (list): Incident beam direction. If None, the
                cross section will be averaged.
            collection_angle (float): Detector collection angle in mrad.
            convergence_angle (float): Beam convergence angle in mrad.
            user_eels_settings (dict): override default EELS config.
                See MPELNESSet.yaml for supported keys.
            nkpts (int): Total number of kpoints in the brillouin zone. Used
                only when feff is run in the reciprocal space mode.
            user_tag_settings (dict): override default tag settings
            **kwargs: Passthrough to FEFFDictSet.
        """
        self.beam_energy = beam_energy
        self.beam_direction = beam_direction
        self.collection_angle = collection_angle
        self.convergence_angle = convergence_angle
        self.user_eels_settings = user_eels_settings
        eels_config_dict = deepcopy(config_dict)
        if beam_direction:
            beam_energy_list = [beam_energy, 0, 1, 1]
            eels_config_dict[spectrum]['BEAM_DIRECTION'] = beam_direction
        else:
            beam_energy_list = [beam_energy, 1, 0, 1]
            del eels_config_dict[spectrum]['BEAM_DIRECTION']
        eels_config_dict[spectrum]['BEAM_ENERGY'] = beam_energy_list
        eels_config_dict[spectrum]['ANGLES'] = [collection_angle, convergence_angle]
        if user_eels_settings:
            eels_config_dict[spectrum].update(user_eels_settings)
        super().__init__(absorbing_atom, structure, radius, eels_config_dict, edge=edge, spectrum=spectrum, nkpts=nkpts, user_tag_settings=user_tag_settings, **kwargs)