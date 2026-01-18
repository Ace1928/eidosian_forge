from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def _compute_cube_index(coords: np.ndarray, global_min: float, radius: float) -> np.ndarray:
    """Compute the cube index from coordinates
    Args:
        coords: (nx3 array) atom coordinates
        global_min: (float) lower boundary of coordinates
        radius: (float) cutoff radius.

    Returns:
        np.ndarray: nx3 array int indices
    """
    return np.array(np.floor((coords - global_min) / radius), dtype=int)