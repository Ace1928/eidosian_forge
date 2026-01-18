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
def _get_reciprocal_point_group(ops, R, A):
    A_inv = np.linalg.inv(A)
    recip_point_group = [np.around(np.dot(A, np.dot(R, A_inv)), decimals=2)]
    for op in ops:
        recip = np.around(np.dot(A, np.dot(op, A_inv)), decimals=2)
        new = True
        new_coset = True
        for thing in recip_point_group:
            if (thing == recip).all():
                new = False
            if (thing == np.dot(R, recip)).all():
                new_coset = False
        if new:
            recip_point_group.append(recip)
        if new_coset:
            recip_point_group.append(np.dot(R, recip))
    return recip_point_group