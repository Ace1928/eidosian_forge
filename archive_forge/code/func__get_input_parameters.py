from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
def _get_input_parameters(self, structure: Structure | Molecule, prev_parameters: dict[str, Any] | None=None) -> dict[str, Any]:
    """Create the input parameters.

        Parameters
        ----------
        structure: Structure | Molecule
            The structure or molecule for the system
        prev_parameters: dict[str, Any]
            The previous calculation's calculation parameters

        Returns:
            dict: The input object
        """
    parameters: dict[str, Any] = {'xc': 'pbe', 'relativistic': 'atomic_zora scalar'}
    prev_parameters = {} if prev_parameters is None else copy.deepcopy(prev_parameters)
    prev_parameters.pop('relax_geometry', None)
    prev_parameters.pop('relax_unit_cell', None)
    kpt_settings = copy.deepcopy(self.user_kpoints_settings)
    if isinstance(structure, Structure) and 'k_grid' in prev_parameters:
        density = self.k2d(structure, prev_parameters.pop('k_grid'))
        if 'density' not in kpt_settings:
            kpt_settings['density'] = density
    parameter_updates = self.get_parameter_updates(structure, prev_parameters)
    parameters = recursive_update(parameters, parameter_updates)
    parameters = recursive_update(parameters, self.user_params)
    if 'k_grid' in parameters and 'density' in kpt_settings:
        warn('WARNING: the k_grid is set in user_params and in the kpt_settings, using the one passed in user_params.', stacklevel=1)
    elif isinstance(structure, Structure) and 'k_grid' not in parameters:
        density = kpt_settings.get('density', 5.0)
        even = kpt_settings.get('even', True)
        parameters['k_grid'] = self.d2k(structure, density, even)
    elif isinstance(structure, Molecule) and 'k_grid' in parameters:
        warn('WARNING: removing unnecessary k_grid information', stacklevel=1)
        del parameters['k_grid']
    return parameters