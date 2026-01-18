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
def _proc_no_rot_sym(self):
    """Handles molecules with no rotational symmetry.

        Only possible point groups are C1, Cs and Ci.
        """
    self.sch_symbol = 'C1'
    if self.is_valid_op(PointGroupAnalyzer.inversion_op):
        self.sch_symbol = 'Ci'
        self.symmops.append(PointGroupAnalyzer.inversion_op)
    else:
        for v in self.principal_axes:
            mirror_type = self._find_mirror(v)
            if mirror_type != '':
                self.sch_symbol = 'Cs'
                break