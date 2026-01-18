from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def _orcc_ab(self):
    prods = self.prods
    orcc_sqr_ab = np.empty(2)
    orcc_sqr_ab[0] = 2.0 * (prods[0] + prods[5])
    orcc_sqr_ab[1] = 2.0 * (prods[1] - prods[5])
    if all(orcc_sqr_ab > 0):
        return np.sqrt(orcc_sqr_ab)