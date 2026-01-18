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
def _proc_asym_top(self):
    """Handles asymmetric top molecules, which cannot contain rotational symmetry
        larger than 2.
        """
    self._check_R2_axes_asym()
    if len(self.rot_sym) == 0:
        logger.debug('No rotation symmetries detected.')
        self._proc_no_rot_sym()
    elif len(self.rot_sym) == 3:
        logger.debug('Dihedral group detected.')
        self._proc_dihedral()
    else:
        logger.debug('Cyclic group detected.')
        self._proc_cyclic()