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
def _get_timing(data):
    begin = False
    user_time, wall_time, sys_time = ([], [], [])
    for row in data:
        splitrow = row.split()
        if 'finished' in splitrow:
            begin = True
        if begin:
            if 'wall' in splitrow:
                wall_time = splitrow[2:10]
            if 'user' in splitrow:
                user_time = splitrow[0:8]
            if 'sys' in splitrow:
                sys_time = splitrow[0:8]
    wall_time_dict = {'h': wall_time[0], 'min': wall_time[2], 's': wall_time[4], 'ms': wall_time[6]}
    user_time_dict = {'h': user_time[0], 'min': user_time[2], 's': user_time[4], 'ms': user_time[6]}
    sys_time_dict = {'h': sys_time[0], 'min': sys_time[2], 's': sys_time[4], 'ms': sys_time[6]}
    return (wall_time_dict, user_time_dict, sys_time_dict)