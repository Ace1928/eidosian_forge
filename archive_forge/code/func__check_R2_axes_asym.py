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
def _check_R2_axes_asym(self):
    """Test for 2-fold rotation along the principal axes.

        Used to handle asymmetric top molecules.
        """
    for v in self.principal_axes:
        op = SymmOp.from_axis_angle_and_translation(v, 180)
        if self.is_valid_op(op):
            self.symmops.append(op)
            self.rot_sym.append((v, 2))