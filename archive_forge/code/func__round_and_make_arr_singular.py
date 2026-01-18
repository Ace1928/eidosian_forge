from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
def _round_and_make_arr_singular(arr: np.ndarray) -> np.ndarray:
    """This function rounds all elements of a matrix to the nearest integer,
    unless the rounding scheme causes the matrix to be singular, in which
    case elements of zero rows or columns in the rounded matrix with the
    largest absolute valued magnitude in the unrounded matrix will be
    rounded to the next integer away from zero rather than to the
    nearest integer.

    The transformation is as follows. First, all entries in 'arr' will be
    rounded to the nearest integer to yield 'arr_rounded'. If 'arr_rounded'
    has any zero rows, then one element in each zero row of 'arr_rounded'
    corresponding to the element in 'arr' of that row with the largest
    absolute valued magnitude will be rounded to the next integer away from
    zero (see the '_round_away_from_zero(x)' function) rather than the
    nearest integer. This process is then repeated for zero columns. Also
    note that if 'arr' already has zero rows or columns, then this function
    will not change those rows/columns.

    Args:
        arr: Input matrix

    Returns:
        Transformed matrix.
    """

    def round_away_from_zero(x):
        """Returns 'x' rounded to the next integer away from 0.
        If 'x' is zero, then returns zero.
        E.g. -1.2 rounds to -2.0. 1.2 rounds to 2.0.
        """
        abs_x = abs(x)
        return math.ceil(abs_x) * (abs_x / x) if x != 0 else 0
    arr_rounded = np.around(arr)
    if (~arr_rounded.any(axis=1)).any():
        zero_row_idxs = np.where(~arr_rounded.any(axis=1))[0]
        for zero_row_idx in zero_row_idxs:
            zero_row = arr[zero_row_idx, :]
            matches = np.absolute(zero_row) == np.amax(np.absolute(zero_row))
            col_idx_to_fix = np.where(matches)[0]
            r_idx = np.random.randint(len(col_idx_to_fix))
            col_idx_to_fix = col_idx_to_fix[r_idx]
            arr_rounded[zero_row_idx, col_idx_to_fix] = round_away_from_zero(arr[zero_row_idx, col_idx_to_fix])
    if (~arr_rounded.any(axis=0)).any():
        zero_col_idxs = np.where(~arr_rounded.any(axis=0))[0]
        for zero_col_idx in zero_col_idxs:
            zero_col = arr[:, zero_col_idx]
            matches = np.absolute(zero_col) == np.amax(np.absolute(zero_col))
            row_idx_to_fix = np.where(matches)[0]
            for idx in row_idx_to_fix:
                arr_rounded[idx, zero_col_idx] = round_away_from_zero(arr[idx, zero_col_idx])
    return arr_rounded.astype(int)