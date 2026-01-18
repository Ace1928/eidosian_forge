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
def _trans_sc_to_Hin(sub_class):
    if sub_class in ['cP1', 'cP2', 'cF1', 'cF2', 'cI1', 'tP1', 'oP1', 'hP1', 'hP2', 'tI1', 'tI2', 'oF1', 'oF3', 'oI1', 'oI3', 'oC1', 'hR1', 'hR2', 'aP1', 'aP2', 'aP3', 'oA1']:
        return np.eye(3)
    if sub_class == 'oF2':
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    if sub_class == 'oI2':
        return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    if sub_class == 'oI3':
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    if sub_class == 'oA2':
        return np.diag((-1, 1, -1))
    if sub_class == 'oC2':
        return np.diag((-1, 1, -1))
    if sub_class in ['mP1', 'mC1', 'mC2', 'mC3']:
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    raise RuntimeError('Sub-classification of crystal not found!')