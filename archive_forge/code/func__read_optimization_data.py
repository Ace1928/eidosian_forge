from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def _read_optimization_data(self):
    if self.data.get('new_optimizer') is None or self.data['version'] == '6':
        temp_energy_trajectory = read_pattern(self.text, {'key': '\\sEnergy\\sis\\s+([\\d\\-\\.]+)'}).get('key')
    else:
        temp_energy_trajectory = read_pattern(self.text, {'key': '\\sStep\\s*\\d+\\s*:\\s*Energy\\s*([\\d\\-\\.]+)'}).get('key')
    if self.data.get('new_optimizer') == [[]] and temp_energy_trajectory is not None:
        temp_energy_trajectory.insert(0, [str(self.data['Total_energy_in_the_final_basis_set'][0])])
    self._read_geometries()
    self._read_gradients()
    if temp_energy_trajectory is None:
        self.data['energy_trajectory'] = []
        if read_pattern(self.text, {'key': 'Error in back_transform'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['back_transform_error']
        elif read_pattern(self.text, {'key': 'pinv\\(\\)\\: svd failed'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['svd_failed']
    else:
        real_energy_trajectory = np.zeros(len(temp_energy_trajectory))
        for ii, entry in enumerate(temp_energy_trajectory):
            real_energy_trajectory[ii] = float(entry[0])
        self.data['energy_trajectory'] = real_energy_trajectory
        if self.data.get('new_optimizer') == [[]]:
            temp_norms = read_pattern(self.text, {'key': 'Norm of Stepsize\\s*([\\d\\-\\.]+)'}).get('key')
            if temp_norms is not None:
                norms = np.zeros(len(temp_norms))
                for ii, val in enumerate(temp_norms):
                    norms[ii] = float(val[0])
                self.data['norm_of_stepsize'] = norms
        if openbabel is not None:
            self.data['structure_change'] = check_for_structure_changes(self.data['initial_molecule'], self.data['molecule_from_last_geometry'])
        if len(self.data.get('errors')) == 0 and self.data.get('optimized_geometry') is None and (len(self.data.get('optimized_zmat')) == 0):
            if read_pattern(self.text, {'key': 'MAXIMUM OPTIMIZATION CYCLES REACHED'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Maximum number of iterations reached during minimization algorithm'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['out_of_opt_cycles']
            elif read_pattern(self.text, {'key': 'UNABLE TO DETERMINE Lamda IN FormD'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['unable_to_determine_lamda']
            elif read_pattern(self.text, {'key': 'Error in back_transform'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['back_transform_error']
            elif read_pattern(self.text, {'key': 'pinv\\(\\)\\: svd failed'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['svd_failed']