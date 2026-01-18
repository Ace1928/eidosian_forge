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
def _get_smallest_set_not_on_axis(self, axis):
    """Returns the smallest list of atoms with the same species and distance from
        origin AND does not lie on the specified axis.

        This maximal set limits the possible rotational symmetry operations, since atoms
        lying on a test axis is irrelevant in testing rotational symmetryOperations.
        """

    def not_on_axis(site):
        v = np.cross(site.coords, axis)
        return np.linalg.norm(v) > self.tol
    valid_sets = []
    _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
    for test_set in dist_el_sites.values():
        valid_set = list(filter(not_on_axis, test_set))
        if len(valid_set) > 0:
            valid_sets.append(valid_set)
    return min(valid_sets, key=len)