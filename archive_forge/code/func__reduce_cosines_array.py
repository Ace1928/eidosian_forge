from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
@staticmethod
def _reduce_cosines_array(orbit_cosines, pop_orbits, pop_labels):
    return [[orb_cos[i] for i in range(len(orb_cos)) if orb_cos[i][0] not in pop_labels] for j, orb_cos in enumerate(orbit_cosines) if j not in pop_orbits]