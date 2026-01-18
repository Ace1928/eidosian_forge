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
@lru_cache(maxsize=32)
def _get_symmetry_dataset(cell, symprec, angle_tolerance):
    """Simple wrapper to cache results of spglib.get_symmetry_dataset since this call is
    expensive.
    """
    return spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)