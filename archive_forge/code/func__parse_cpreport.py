from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
def _parse_cpreport(self, cpreport):

    def get_type(signature: int, is_nucleus: bool):
        if signature == 3:
            return 'cage'
        if signature == 1:
            return 'ring'
        if signature == -1:
            return 'bond'
        if signature == -3:
            if is_nucleus:
                return 'nucleus'
            return 'nnattr'
        return None
    bohr_to_angstrom = 0.529177
    self.critical_points = [CriticalPoint(p['id'] - 1, get_type(p['signature'], p['is_nucleus']), p['fractional_coordinates'], p['point_group'], p['multiplicity'], p['field'], p['gradient'], coords=[x * bohr_to_angstrom for x in p['cartesian_coordinates']] if cpreport['units'] == 'bohr' else None, field_hessian=p['hessian']) for p in cpreport['critical_points']['nonequivalent_cps']]
    for point in cpreport['critical_points']['cell_cps']:
        self._add_node(idx=point['id'] - 1, unique_idx=point['nonequivalent_id'] - 1, frac_coords=point['fractional_coordinates'])
        if 'attractors' in point:
            self._add_edge(idx=point['id'] - 1, from_idx=int(point['attractors'][0]['cell_id']) - 1, from_lvec=point['attractors'][0]['lvec'], to_idx=int(point['attractors'][1]['cell_id']) - 1, to_lvec=point['attractors'][1]['lvec'])