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
def _read_scan_data(self):
    temp_energy_trajectory = read_pattern(self.text, {'key': '\\sEnergy\\sis\\s+([\\d\\-\\.]+)'}).get('key')
    if temp_energy_trajectory is None:
        self.data['energy_trajectory'] = []
    else:
        real_energy_trajectory = np.zeros(len(temp_energy_trajectory))
        for ii, entry in enumerate(temp_energy_trajectory):
            real_energy_trajectory[ii] = float(entry[0])
        self.data['energy_trajectory'] = real_energy_trajectory
    self._read_geometries()
    if openbabel is not None:
        self.data['structure_change'] = check_for_structure_changes(self.data['initial_molecule'], self.data['molecule_from_last_geometry'])
    self._read_gradients()
    if len(self.data.get('errors')) == 0:
        if read_pattern(self.text, {'key': 'MAXIMUM OPTIMIZATION CYCLES REACHED'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['out_of_opt_cycles']
        elif read_pattern(self.text, {'key': 'UNABLE TO DETERMINE Lamda IN FormD'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['unable_to_determine_lamda']
    header_pattern = '\\s*\\-+ Summary of potential scan\\: \\-+\\s*'
    row_pattern_single = '\\s*([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n'
    row_pattern_double = '\\s*([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n'
    footer_pattern = '\\s*\\-+'
    single_data = read_table_pattern(self.text, header_pattern=header_pattern, row_pattern=row_pattern_single, footer_pattern=footer_pattern)
    self.data['scan_energies'] = []
    if len(single_data) == 0:
        double_data = read_table_pattern(self.text, header_pattern=header_pattern, row_pattern=row_pattern_double, footer_pattern=footer_pattern)
        if len(double_data) == 0:
            self.data['scan_energies'] = None
        else:
            for line in double_data[0]:
                params = [float(line[0]), float(line[1])]
                energy = float(line[2])
                self.data['scan_energies'].append({'params': params, 'energy': energy})
    else:
        for line in single_data[0]:
            param = float(line[0])
            energy = float(line[1])
            self.data['scan_energies'].append({'params': param, 'energy': energy})
    scan_inputs_head = '\\s*\\$[Ss][Cc][Aa][Nn]'
    scan_inputs_row = '\\s*([Ss][Tt][Rr][Ee]|[Tt][Oo][Rr][Ss]|[Bb][Ee][Nn][Dd]) '
    scan_inputs_row += '((?:[0-9]+\\s+)+)([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*'
    scan_inputs_foot = '\\s*\\$[Ee][Nn][Dd]'
    constraints_meta = read_table_pattern(self.text, header_pattern=scan_inputs_head, row_pattern=scan_inputs_row, footer_pattern=scan_inputs_foot)
    self.data['scan_variables'] = {'stre': [], 'bend': [], 'tors': []}
    for row in constraints_meta[0]:
        var_type = row[0].lower()
        self.data['scan_variables'][var_type].append({'atoms': [int(i) for i in row[1].split()], 'start': float(row[2]), 'end': float(row[3]), 'increment': float(row[4])})
    temp_constraint = read_pattern(self.text, {'key': '\\s*(Distance\\(Angs\\)|Angle|Dihedral)\\:\\s*((?:[0-9]+\\s+)+)+([\\.0-9]+)\\s+([\\.0-9]+)'}).get('key')
    self.data['scan_constraint_sets'] = {'stre': [], 'bend': [], 'tors': []}
    if temp_constraint is not None:
        for entry in temp_constraint:
            atoms = [int(i) for i in entry[1].split()]
            current = float(entry[2])
            target = float(entry[3])
            if entry[0] == 'Distance(Angs)':
                if len(atoms) == 2:
                    self.data['scan_constraint_sets']['stre'].append({'atoms': atoms, 'current': current, 'target': target})
            elif entry[0] == 'Angle':
                if len(atoms) == 3:
                    self.data['scan_constraint_sets']['bend'].append({'atoms': atoms, 'current': current, 'target': target})
            elif entry[0] == 'Dihedral' and len(atoms) == 4:
                self.data['scan_constraint_sets']['tors'].append({'atoms': atoms, 'current': current, 'target': target})