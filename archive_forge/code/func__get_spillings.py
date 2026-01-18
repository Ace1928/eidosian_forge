from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
@staticmethod
def _get_spillings(data, number_of_spins):
    charge_spilling = []
    total_spilling = []
    for row in data:
        splitrow = row.split()
        if len(splitrow) > 2 and splitrow[2] == 'spilling:':
            if splitrow[1] == 'charge':
                charge_spilling += [np.float64(splitrow[3].replace('%', '')) / 100.0]
            if splitrow[1] == 'total':
                total_spilling += [np.float64(splitrow[3].replace('%', '')) / 100.0]
        if len(charge_spilling) == number_of_spins and len(total_spilling) == number_of_spins:
            break
    return (charge_spilling, total_spilling)