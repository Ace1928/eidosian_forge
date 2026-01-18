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
def _three_to_one(label3d: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """The reverse of _one_to_three."""
    return np.array(label3d[:, 0] * ny * nz + label3d[:, 1] * nz + label3d[:, 2]).reshape((-1, 1))