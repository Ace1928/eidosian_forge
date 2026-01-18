from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def _is_in_miller_family(miller_index: tuple[int, int, int], miller_list: list[tuple[int, int, int]], symm_ops: list) -> bool:
    """Helper function to check if the given Miller index belongs
    to the same family of any index in the provided list.

    Args:
        miller_index (tuple): The Miller index to analyze.
        miller_list (list): List of Miller indices.
        symm_ops (list): Symmetry operations for a lattice,
            used to define the indices family.
    """
    return any((in_coord_list(miller_list, op.operate(miller_index)) for op in symm_ops))