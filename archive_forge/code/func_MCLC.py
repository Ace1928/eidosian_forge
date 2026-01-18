from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def MCLC(self):
    orcc_ab = self._orcc_ab()
    if orcc_ab is None:
        return None
    prods = self.prods
    C = self.C
    mclc_a, mclc_b = orcc_ab[::-1]
    mclc_cosa = 2.0 * prods[3] / (mclc_b * C)
    if -1 < mclc_cosa < 1:
        mclc_alpha = np.arccos(mclc_cosa) * 180 / np.pi
        if mclc_b > C:
            mclc_b = 0.5 * (mclc_b + C)
            C = mclc_b
        return self._check(MCLC, mclc_a, mclc_b, C, mclc_alpha)