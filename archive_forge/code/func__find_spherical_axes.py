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
def _find_spherical_axes(self):
    """Looks for R5, R4, R3 and R2 axes in spherical top molecules.

        Point group T molecules have only one unique 3-fold and one unique 2-fold axis. O
        molecules have one unique 4, 3 and 2-fold axes. I molecules have a unique 5-fold
        axis.
        """
    rot_present = defaultdict(bool)
    _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
    test_set = min(dist_el_sites.values(), key=len)
    coords = [s.coords for s in test_set]
    for c1, c2, c3 in itertools.combinations(coords, 3):
        for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
            if not rot_present[2]:
                test_axis = cc1 + cc2
                if np.linalg.norm(test_axis) > self.tol:
                    op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                    rot_present[2] = self.is_valid_op(op)
                    if rot_present[2]:
                        self.symmops.append(op)
                        self.rot_sym.append((test_axis, 2))
        test_axis = np.cross(c2 - c1, c3 - c1)
        if np.linalg.norm(test_axis) > self.tol:
            for r in (3, 4, 5):
                if not rot_present[r]:
                    op = SymmOp.from_axis_angle_and_translation(test_axis, 360 / r)
                    rot_present[r] = self.is_valid_op(op)
                    if rot_present[r]:
                        self.symmops.append(op)
                        self.rot_sym.append((test_axis, r))
                        break
        if rot_present[2] and rot_present[3] and (rot_present[4] or rot_present[5]):
            break