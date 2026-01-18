from __future__ import annotations
import os
import sqlite3
import textwrap
from array import array
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from monty.design_patterns import cached_class
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.groups import SymmetryGroup, in_array_list
from pymatgen.symmetry.settings import JonesFaithfulTransformation
from pymatgen.util.string import transformation_to_string
def _get_point_operator(idx):
    """Retrieve information on point operator (rotation matrix and Seitz label)."""
    is_hex = self._data['bns_number'][0] >= 143 and self._data['bns_number'][0] <= 194
    c.execute('SELECT symbol, matrix FROM point_operators WHERE idx=? AND hex=?;', (idx - 1, is_hex))
    op = c.fetchone()
    return {'symbol': op[0], 'matrix': np.array(op[1].split(','), dtype='f').reshape(3, 3)}