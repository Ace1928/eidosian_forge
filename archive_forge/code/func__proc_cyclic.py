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
def _proc_cyclic(self):
    """Handles cyclic group molecules."""
    main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
    self.sch_symbol = f'C{rot}'
    mirror_type = self._find_mirror(main_axis)
    if mirror_type == 'h':
        self.sch_symbol += 'h'
    elif mirror_type == 'v':
        self.sch_symbol += 'v'
    elif mirror_type == '' and self.is_valid_op(SymmOp.rotoreflection(main_axis, angle=180 / rot)):
        self.sch_symbol = f'S{2 * rot}'