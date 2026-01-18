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
def _read_species_and_inital_geometry(self):
    """Parses species and initial geometry."""
    header_pattern = 'Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
    table_pattern = '\\s*\\d+\\s+([a-zA-Z]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
    footer_pattern = '\\s*-+'
    temp_geom = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
    if temp_geom is None or len(temp_geom) == 0:
        self.data['species'] = self.data['initial_geometry'] = self.data['initial_molecule'] = self.data['point_group'] = None
    else:
        temp_point_group = read_pattern(self.text, {'key': 'Molecular Point Group\\s+([A-Za-z\\d\\*]+)'}, terminate_on_match=True).get('key')
        if temp_point_group is not None:
            self.data['point_group'] = temp_point_group[0][0]
        else:
            self.data['point_group'] = None
        temp_geom = temp_geom[0]
        species = []
        geometry = np.zeros(shape=(len(temp_geom), 3), dtype=float)
        for ii, entry in enumerate(temp_geom):
            species += [entry[0]]
            for jj in range(3):
                if '*' in entry[jj + 1]:
                    geometry[ii, jj] = 10000000000.0
                else:
                    geometry[ii, jj] = float(entry[jj + 1])
        self.data['species'] = species
        self.data['initial_geometry'] = geometry
        if self.data['charge'] is not None and self.data['multiplicity'] is not None:
            self.data['initial_molecule'] = Molecule(species=species, coords=geometry, charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
        else:
            self.data['initial_molecule'] = None