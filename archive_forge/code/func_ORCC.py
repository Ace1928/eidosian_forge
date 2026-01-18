from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def ORCC(self):
    orcc_lengths_ab = self._orcc_ab()
    if orcc_lengths_ab is None:
        return None
    return self._check(ORCC, *orcc_lengths_ab, self.C)