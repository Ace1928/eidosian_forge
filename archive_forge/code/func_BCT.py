from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def BCT(self):
    lengths = self._bct_orci_lengths()
    if lengths is None:
        return None
    return self._check(BCT, lengths[0], lengths[2])