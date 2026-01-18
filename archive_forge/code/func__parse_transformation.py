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
def _parse_transformation(b):
    """Parses compact binary representation into transformation between OG and BNS settings."""
    if len(b) == 0:
        return None
    P = [[b[0], b[3], b[6]], [b[1], b[4], b[7]], [b[2], b[5], b[8]]]
    p = [b[9] / b[12], b[10] / b[12], b[11] / b[12]]
    P = np.array(P).transpose()
    P_string = transformation_to_string(P, components=('a', 'b', 'c'))
    p_string = f'{Fraction(p[0]).limit_denominator()},{Fraction(p[1]).limit_denominator()},{Fraction(p[2]).limit_denominator()}'
    return P_string + ';' + p_string