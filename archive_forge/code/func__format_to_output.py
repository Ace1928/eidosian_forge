from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
@staticmethod
def _format_to_output(tensor, tensor_doping, output, doping_levels, multi=1.0):
    if doping_levels:
        full_tensor = tensor_doping
        result = {doping: {t: [] for t in tensor_doping[doping]} for doping in tensor_doping}
        for doping in full_tensor:
            for temp in full_tensor[doping]:
                for i in range(len(full_tensor[doping][temp])):
                    if output in ['eig', 'eigs']:
                        result[doping][temp].append(sorted(np.linalg.eigh(full_tensor[doping][temp][i])[0] * multi))
                    elif output == 'tensor':
                        result[doping][temp].append(np.array(full_tensor[doping][temp][i]) * multi)
                    elif output == 'average':
                        result[doping][temp].append((full_tensor[doping][temp][i][0][0] + full_tensor[doping][temp][i][1][1] + full_tensor[doping][temp][i][2][2]) * multi / 3.0)
                    else:
                        raise ValueError(f'Unknown output format: {output}')
    else:
        full_tensor = tensor
        result = {t: [] for t in tensor}
        for temp in full_tensor:
            for i in range(len(tensor[temp])):
                if output in ['eig', 'eigs']:
                    result[temp].append(sorted(np.linalg.eigh(full_tensor[temp][i])[0] * multi))
                elif output == 'tensor':
                    result[temp].append(np.array(full_tensor[temp][i]) * multi)
                elif output == 'average':
                    result[temp].append((full_tensor[temp][i][0][0] + full_tensor[temp][i][1][1] + full_tensor[temp][i][2][2]) * multi / 3.0)
                else:
                    raise ValueError(f'Unknown output format: {output}')
    return result