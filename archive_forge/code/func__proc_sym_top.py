from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def _proc_sym_top(self):
    """Handles symmetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.

        More complex handling required to look for R2 axes perpendicular to this unique
        axis.
        """
    if abs(self.eigvals[0] - self.eigvals[1]) < self.eig_tol:
        ind = 2
    elif abs(self.eigvals[1] - self.eigvals[2]) < self.eig_tol:
        ind = 0
    else:
        ind = 1
    logger.debug(f'Eigenvalues = {self.eigvals}.')
    unique_axis = self.principal_axes[ind]
    self._check_rot_sym(unique_axis)
    logger.debug(f'Rotation symmetries = {self.rot_sym}')
    if len(self.rot_sym) > 0:
        self._check_perpendicular_r2_axis(unique_axis)
    if len(self.rot_sym) >= 2:
        self._proc_dihedral()
    elif len(self.rot_sym) == 1:
        self._proc_cyclic()
    else:
        self._proc_no_rot_sym()