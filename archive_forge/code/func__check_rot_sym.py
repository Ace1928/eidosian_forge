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
def _check_rot_sym(self, axis):
    """Determines the rotational symmetry about supplied axis.

        Used only for symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
    min_set = self._get_smallest_set_not_on_axis(axis)
    max_sym = len(min_set)
    for i in range(max_sym, 0, -1):
        if max_sym % i != 0:
            continue
        op = SymmOp.from_axis_angle_and_translation(axis, 360 / i)
        rotvalid = self.is_valid_op(op)
        if rotvalid:
            self.symmops.append(op)
            self.rot_sym.append((axis, i))
            return i
    return 1